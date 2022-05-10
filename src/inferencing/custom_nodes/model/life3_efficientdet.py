"""
Node template for creating custom nodes.
"""

from pconst import const
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from peekingduck.pipeline.nodes.node import AbstractNode
from inferencing.custom_nodes.model.life3_efficientdet_model import (
    Life3EfficientDetModel,
)


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        pkd_base_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            pkd_base_dir=Path(const.PKD_BASE_DIR).resolve(),
            node_path=__name__,
            **kwargs,
        )

        self.model = Life3EfficientDetModel(self.config)

        # load model
        self.loaded_model = self.model.load_model()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does prediction using the
        fine-tuned life3 efficientdet model.
        """

        # make predictions
        outputs = self.model.predict(
            # model=model, image=inputs["img"], filename=inputs["filename"]  # remove filename - move output file handling to sahi
            model=self.loaded_model,
            image=inputs["img"],
        )
        return outputs
