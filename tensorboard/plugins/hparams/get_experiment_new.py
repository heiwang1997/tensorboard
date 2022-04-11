# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Classes and functions for handling the GetExperiment API call."""
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format


class Handler(object):
    """Handles a GetExperiment request."""

    def __init__(self, request_context, backend_context, experiment_id):
        """Constructor.

        Args:
          request_context: A tensorboard.context.RequestContext.
          backend_context: A backend_context.Context instance.
          experiment_id: A string, as from `plugin_util.experiment_id`.
        """
        self._request_context = request_context
        self._backend_context = backend_context
        self._experiment_id = experiment_id
        self._hparams_run_to_tag_to_content = backend_context.hparams_metadata(
            request_context, experiment_id
        )
        # Since an context.experiment() call may search through all the runs, we
        # cache it here.
        self._experiment = backend_context.experiment_from_metadata(
            request_context, experiment_id, self._hparams_run_to_tag_to_content
        )

        self._ih_set = self._inspect_hparams()

    def _inspect_hparams(self):
        ref_hparam = None
        ih_set = set()
        for (
            session_name,
            tag_to_content,
        ) in self._hparams_run_to_tag_to_content.items():
            if metadata.SESSION_START_INFO_TAG not in tag_to_content:
                continue
            start_info = metadata.parse_session_start_info_plugin_data(
                tag_to_content[metadata.SESSION_START_INFO_TAG]
            )
            hparam_dict = json_format.MessageToDict(start_info)['hparams']
            if ref_hparam is None:
              ref_hparam = hparam_dict
            else:
              for h_key in hparam_dict.keys():
                if h_key not in ref_hparam.keys() or ref_hparam[h_key] != hparam_dict[h_key]:
                  ih_set.add(h_key)
        return ih_set

    def run(self):
        """Handles the request specified on construction.

        Returns:
          An Experiment object.
        """
        return self._experiment, self._ih_set
