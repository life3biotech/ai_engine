"""
Node template for creating custom nodes.
"""

from pconst import const
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from peekingduck.pipeline.nodes.node import AbstractNode
from src.inference.custom_nodes.model.life3_efficientdet_model import (
    Life3EfficientDetModel,
)
from src.life3_biotech.config import PipelineConfig
from src.life3_biotech import general_utils


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

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does prediction using the
        fine-tuned life3 efficientdet model.
        """

        # load model
        model = self.model.load_model()

        # make predictions
        outputs = self.model.predict(
            model=model, image=inputs["img"], filename=inputs["filename"]
        )
        return outputs
