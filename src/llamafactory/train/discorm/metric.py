# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict, List

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""Compute reward accuracy. Optionally computes variance metrics if available."""
    # Initialize with accuracy key, variance keys can be added dynamically
    score_dict: Dict[str, List[float]] = field(default_factory=lambda: {"accuracy": []})

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if self.score_dict:
            # Calculate mean only for metrics that have been populated
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items() if v}

        # Reset score_dict for the next evaluation
        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        # Ensure score_dict is initialized if it wasn't by default_factory (e.g., if loaded)
        if not isinstance(self.score_dict, dict) or "accuracy" not in self.score_dict:
             self.score_dict = {"accuracy": []}
        # Initialize variance keys if they don't exist, but keep lists empty
        self.score_dict.setdefault("mean_chosen_variance", [])
        self.score_dict.setdefault("mean_rejected_variance", [])

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        # Core logic: Compute accuracy based on the first two predictions (scores/means)
        chosen_scores = numpify(eval_preds.predictions[0])
        rejected_scores = numpify(eval_preds.predictions[1])

        if not chosen_scores.shape:
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        # Optional: Compute variance metrics if available
        if len(eval_preds.predictions) >= 4:
            chosen_vars = numpify(eval_preds.predictions[2])
            rejected_vars = numpify(eval_preds.predictions[3])
            
            # Ensure variance keys exist before appending
            self.score_dict.setdefault("mean_chosen_variance", [])
            self.score_dict.setdefault("mean_rejected_variance", [])

            if not chosen_vars.shape:
                self.score_dict["mean_chosen_variance"].append(chosen_vars)
                self.score_dict["mean_rejected_variance"].append(rejected_vars)
            else:
                for i in range(len(chosen_vars)):
                    self.score_dict["mean_chosen_variance"].append(chosen_vars[i])
                    self.score_dict["mean_rejected_variance"].append(rejected_vars[i])

        if compute_result:
            return self._dump()
        else:
            return None # Still accumulating
