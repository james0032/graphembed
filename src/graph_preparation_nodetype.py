import pandas as pd
import polars as pl
import os
import argparse


def merge(args):
    

    nodes = pl.read_csv(args.nodein, separator="\t")
    print("Starting node numbers:",nodes.shape)
    #leftnodes = pd.read_csv("//projects/aixb/jchung/everycure/KGs/baseline2_leftnodes.list", header=None, names=["curie"])
    edges = pl.read_csv(args.edgein, separator="\t")
    print("Starting edge numbers:", edges.shape)

    edges = edges.join(
    nodes.select(["id", "category"]),
    left_on="subject",
    right_on="id",
    how="left"
    ).rename({"category": "subject_category"})
    
    edges = edges.join(
    nodes.select(["id", "category"]),
    left_on="object",
    right_on="id",
    how="left"
    ).rename({"category": "object_category"})

    
    
    pattern = "|".join(args.keys.replace(" ", "").split(","))
    
    edges = edges.with_columns(pl.col("subject_category").str.contains(pattern).alias("subject_in_keys"))
    edges = edges.with_columns(pl.col("object_category").str.contains(pattern).alias("object_in_keys"))
    
    edges = edges.filter(pl.col("subject_in_keys") & pl.col("object_in_keys"))
    
    
    
    # drop those category mapping and boolean columns
    edges_dropped = edges.drop(["subject_category", "object_category", "subject_in_keys", "object_in_keys"])
    
    edges_dropped.write_csv(args.edgeout, separator="\t")
    print("left edge numbers", edges_dropped.shape)
    # orphan node removal
    sub =  edges.select("subject").unique()["subject"].to_list()
    ob  =  edges.select("object").unique()["object"].to_list()
    leftnodes = sub + ob
    nodes = nodes.filter(pl.col("id").is_in(leftnodes))
    print("left node numbers", nodes.shape)
    nodes.write_csv(args.nodeout, separator="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, help="How to select subset of knowledge graph", default="keep_CCDD")
    parser.add_argument("--nodein", type=str, help="Input node file path")
    parser.add_argument("--edgein", type=str, help="Input edge file path")
    parser.add_argument("--nodeout", type=str, help="Output node file path")
    parser.add_argument("--edgeout", type=str, help="Output edge file path")
    parser.add_argument("--keys", type=str, help="This is going to be a list of node types as filter, has to be separated by , for each node type without whitespace")
    args = parser.parse_args()
    merge(args)