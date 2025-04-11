from graphviz import Digraph
from mlframe.data.reader import IReader
from mlframe.data.transformer import ITransformer
from mlframe.data.writer import IWriter
import warnings


class DatasetPipeline:
    def __init__(
        self, reader: IReader, transformer: ITransformer = None, writer: IWriter = None
    ):
        self.reader = reader
        self.transformer = transformer
        self.writer = writer

    def __call__(
        self,
    ):
        print(">>[INFO] Reading dataset")
        datasets = self.reader()

        if self.transformer is not None:
            print(">>[INFO] Transforming dataset")
            datasets = self.transformer(datasets)

        if self.writer is not None:
            print(">>[INFO] Writing dataset")
            datasets = self.writer(datasets)

        return datasets

    def plot_graph(self, save: bool = False):
        # TODO: plot_graph for readers, transformers and writers
        pipeline_dot = Digraph(
            comment="Data Pipeline", graph_attr={"rankdir": "LR"}, format="png"
        )

        reader_dot = Digraph(
            name="cluster_reader", comment="Reader", graph_attr={"rankdir": "LR"}
        )
        reader_dot = self.reader.plot_graph(reader_dot)
        reader_dot.attr(label="Reader", labelloc="t", fontsize="14", fontweight="bold")
        reader_dot.attr(style="filled", color="black", fillcolor="lightgreen")
        pipeline_dot.subgraph(reader_dot)

        transformer_dot = Digraph(
            name="cluster_transformer",
            comment="Transformer",
            graph_attr={"rankdir": "LR"},
        )
        transformer_dot = self.transformer.plot_graph(transformer_dot)
        transformer_dot.attr(
            label="Transformer", labelloc="t", fontsize="14", fontweight="bold"
        )
        transformer_dot.attr(style="filled", color="black", fillcolor="lightyellow")
        pipeline_dot.subgraph(transformer_dot)

        writer_dot = Digraph(
            name="cluster_writer", comment="Writer", graph_attr={"rankdir": "LR"}
        )
        writer_dot = self.writer.plot_graph(writer_dot)
        writer_dot.attr(label="Writer", labelloc="t", fontsize="14", fontweight="bold")
        writer_dot.attr(style="filled", color="black", fillcolor="lightred")
        pipeline_dot.subgraph(writer_dot)
        if save:
            pipeline_dot.render("pipeline")
        return pipeline_dot.pipe()

    def set_transformer_stopper(self, *args, **kwargs):
        "Helping method to set a stopper in transformer step for debugging purposes"
        if hasattr(self.transformer, "set_stopper"):
            self.transformer.set_stopper(*args, **kwargs)
        else:
            warnings.warn("Transformer does not support set_stopper")
        return self
