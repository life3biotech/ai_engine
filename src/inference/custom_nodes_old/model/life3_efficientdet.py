"""
Node template for creating custom nodes.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode
from src.inference.custom_nodes.model.life3_efficientdet_model import (
    Life3EfficientDetModel,
)


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.model = Life3EfficientDetModel(self.config)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        outputs = self.model.main(image=inputs["img"])
        # bboxes = np.clip(bboxes, 0, 1)
        # outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        return outputs
