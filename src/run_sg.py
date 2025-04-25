import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use memory growth (recommended for better GPU utilization)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is being used for TensorFlow.")
    except RuntimeError as e:
        print(e)
import stellargraph as sg
from tensorflow import keras
import json
import pandas as pd
import numpy as np
import os
import argparse
import pickle
from stellargraph.data import UnsupervisedSampler



class BatchLossCallBack(keras.callbacks.Callback):
    def __init__(self, save_path, save_every_n_batches=100, save_until = 500):
        super().__init__()
        self.batch_losses_per_epoch = {} # Dict to store batch losses by epoch
        self.save_path = save_path
        self.save_every_n_batches= save_every_n_batches
        self.save_until = save_until
    
    def on_epoch_begin(self, epoch, logs=None):
        """Initialize storage for the new epoch"""
        self.current_epoch  = epoch
        self.batch_losses_per_epoch[epoch] = []
        
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses_per_epoch[self.current_epoch].append(logs['loss'])
        
        if ((batch) % self.save_every_n_batches == 0) and (batch < self.save_until):
            save_file = os.path.join(self.save_path, f"batch_{batch+1}.keras")
            self.model.save(save_file)
            print(f"Model saved at batch {batch+1}")

def             

def exec(args):
    
    edge_df = pd.read_csv(os.path.join(os.getcwd(), args.style , "filtered_graph_edges.txt"), sep="\t", usecols = ['source', 'target'])
    edge_df = edge_df.drop_duplicates().reset_index(drop=True)
    node_df = pd.read_json(os.path.join(os.getcwd(), args.style , "filtered_graph_nodes_info.txt"), lines=True)
    node_df = node_df.drop_duplicates(subset="id")
    # Generate initial feature tensor for input nodes
    ntf = tf.random.uniform((node_df.shape[0], args.dim), minval=0, maxval=1, dtype=tf.float32, seed=41)
    feat_df = pd.DataFrame(ntf.numpy())
    feat_df.set_index(node_df["id"], inplace=True)

    # Build networkx graph
    robokop_stellargraph = sg.StellarGraph(nodes = feat_df, edges = edge_df)

    nodes = list(robokop_stellargraph.nodes())

    # Set samples for unsupervised learning
    unsupervised_samples = UnsupervisedSampler(
        robokop_stellargraph, nodes=nodes, length=args.walk_length, number_of_walks=args.number_of_walks
    )
    # Set GraphSAGE training parameters
    num_samples = [25, 10]
    generator = sg.mapper.GraphSAGELinkGenerator(robokop_stellargraph, args.batch_size, num_samples)
    layer_sizes = [args.dim, args.dim]
    
    
    graphsage = sg.layer.GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )

    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = sg.layer.link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    exec_callback = BatchLossCallBack(save_path=os.path.join(os.getcwd(), args.style),
                                    save_every_n_batches=args.save_every_n_batches,
                                    save_until=(args.save_every_n_batches*5)
                                    )
    
    train_gen = generator.flow(unsupervised_samples)

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        verbose=1,
        use_multiprocessing=True,
        workers=18,
        shuffle=False,
        callbacks=[exec_callback]
    )
    
    # Save final model
    model_file = os.path.join(os.getcwd(), args.style , "final_model.keras") 
    model.save(model_file)
    
    # Build model to acquire node embedding
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs = x_out_src)
    gen = sg.mapper.GraphSAGENodeGenerator(robokop_stellargraph, args.batch_size, num_samples)
    node_gen = gen.flow(nodes)
    allne = np.empty(shape=[0,args.dim])
    for g, _ in node_gen:
        node_embeddings = embedding_model.predict(g)
        allne = np.vstack([allne, node_embeddings])

    # Assign output file paths

    unsupervised_graphsage_emb_dict = {}
    embed_pkl = os.path.join(os.getcwd(), args.style, args.style+"_node_embed.pkl")
    for (nodeid, nodeemb) in zip(nodes, allne):
        unsupervised_graphsage_emb_dict[nodeid] = nodeemb
    with open(embed_pkl, "wb") as empkl:
        pickle.dump(unsupervised_graphsage_emb_dict, empkl) 

    embed_idx = os.path.join(os.getcwd(), args.style, args.style+"_node_embed.idx")
    with open(embed_idx, "w") as idxfile:
        idxfile.write("\n".join(nodes))
    
    batch_loss_file = os.path.join(os.getcwd(), args.style, "batch_loss.json")
    with open(batch_loss_file, "w") as lossfile:
        json.dump(exec_callback.batch_losses_per_epoch, lossfile, indent=4)
        
    history_file = os.path.join(os.getcwd(), args.style, "history.json")
    with open(history_file, "w") as hisfile:
        json.dump(history.history, hisfile, indent=4)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, help="How to select subset of knowledge graph", default="keep_CCDD")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--number_of_walks", type=int, default=5)
    parser.add_argument("--walk_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every_n_batches", type=int, default=100)
    args = parser.parse_args()
    exec(args)