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
"""The TensorBoard HParams plugin.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""


import json
import os
import time

import werkzeug
from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context
from tensorboard.plugins.hparams import download_data
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import get_experiment_new as get_experiment
from tensorboard.plugins.hparams import list_metric_evals
from tensorboard.plugins.hparams import list_session_groups
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.util import tb_logging
from pathlib import Path


logger = tb_logging.get_logger()


class HParamsPlugin(base_plugin.TBPlugin):
    """HParams Plugin for TensorBoard.

    It supports both GETs and POSTs. See 'http_api.md' for more details.
    """

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates HParams plugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._context = backend_context.Context(context)

    def get_plugin_apps(self):
        """See base class."""

        return {
            "/download_data": self.download_data_route,
            "/experiment": self.get_experiment_route,
            "/session_groups": self.list_session_groups_route,
            "/metric_evals": self.list_metric_evals_route,
            "/comment_get": self.comment_get_route,
            "/comment_update": self.comment_update_route,
            "/run_info": self.run_info_route,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-hparams-dashboard")

    # ---- /download_data- -------------------------------------------------------
    @wrappers.Request.application
    def download_data_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            response_format = request.args.get("format")
            columns_visibility = json.loads(
                request.args.get("columnsVisibility")
            )
            request_proto = _parse_request_argument(
                request, api_pb2.ListSessionGroupsRequest
            )
            session_groups = list_session_groups.Handler(
                ctx, self._context, experiment_id, request_proto
            ).run()
            experiment = get_experiment.Handler(
                ctx, self._context, experiment_id
            ).run()
            body, mime_type = download_data.Handler(
                self._context,
                experiment,
                session_groups,
                response_format,
                columns_visibility,
            ).run()
            return http_util.Respond(request, body, mime_type)
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /experiment -----------------------------------------------------------
    # This will be called only once on page loaded, to obtain hyperparameter names.
    #   We modify it, to also return whether this hparam is modified.
    @wrappers.Request.application
    def get_experiment_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            # This backend currently ignores the request parameters, but (for a POST)
            # we must advance the input stream to skip them -- otherwise the next HTTP
            # request will be parsed incorrectly.
            _ = _parse_request_argument(request, api_pb2.GetExperimentRequest)
            exp_info, ih_info = get_experiment.Handler(ctx, self._context, experiment_id).run()
            exp_json = json_format.MessageToDict(
                exp_info,
                including_default_value_fields=True,
            )
            for ih_name in ih_info:
                for t in exp_json['hparamInfos']:
                    if t['name'] == ih_name:
                        t['diff'] = True
                        break
            return http_util.Respond(
                request,
                exp_json,
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /session_groups -------------------------------------------------------
    # This will be called everytime a filter / new hparams is applied / ticked, so that
    #   the corresponding values will be queried.
    @wrappers.Request.application
    def list_session_groups_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(
                request, api_pb2.ListSessionGroupsRequest
            )
            return http_util.Respond(
                request,
                json_format.MessageToJson(
                    list_session_groups.Handler(
                        ctx, self._context, experiment_id, request_proto
                    ).run(),
                    including_default_value_fields=True,
                ),
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /metric_evals ---------------------------------------------------------
    @wrappers.Request.application
    def list_metric_evals_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(
                request, api_pb2.ListMetricEvalsRequest
            )
            scalars_plugin = self._get_scalars_plugin()
            if not scalars_plugin:
                raise werkzeug.exceptions.NotFound("Scalars plugin not loaded")
            return http_util.Respond(
                request,
                json.dumps(
                    list_metric_evals.Handler(
                        ctx, request_proto, scalars_plugin, experiment_id
                    ).run()
                ),
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /comment_get ---------------------------------------------------------
    @wrappers.Request.application
    def comment_get_route(self, request):
        sg_name = json.loads(request.data)['name']
        comment_dir = Path(self._context.tb_context.logdir) / sg_name
        try:
            with (comment_dir / "comment.txt").open("r") as f:
                comment_data = f.read()
        except FileNotFoundError:
            comment_data = ""
        is_done = (comment_dir / "done").exists()
        return http_util.Respond(
            request,
            json.dumps(
                {"value": comment_data, "done": is_done}
            ),
            "application/json",
        )

    # ---- /comment_update ---------------------------------------------------------
    @wrappers.Request.application
    def comment_update_route(self, request):
        update_info = json.loads(request.data)
        sg_name = update_info['name']
        comment_dir = Path(self._context.tb_context.logdir) / sg_name
        with (comment_dir / "comment.txt").open("w") as f:
            f.write(update_info['value'])
        if update_info['done']:
            with (comment_dir / "done").open("w") as f:
                f.write("1")
        else:
            try:
                os.remove(str(comment_dir / "done"))
            except OSError:
                pass
        return http_util.Respond(request, json.dumps({}), "application/json",)

    # ---- /run_info ---------------------------------------------------------
    @wrappers.Request.application
    def run_info_route(self, request):
        sg_name = json.loads(request.data)['name']
        exp_dir = Path(self._context.tb_context.logdir) / sg_name
        event_mtimes = [(pth, pth.lstat().st_mtime) for pth in exp_dir.glob("events.out*")]
        # Sort all events files according to modified time, and take the most recent one.
        event_mtimes = sorted(event_mtimes, key=lambda t: t[1], reverse=True)[0]

        # Get host name and time differences.
        hostname = event_mtimes[0].name.split(".")[4]
        time_diff_hrs = (time.time() - event_mtimes[1]) / 3600

        # Let's see if we have a better displayed name for hosts.
        try:
            with open(os.environ['JH_UTIL_DIR'] + "/synch/alternative_paths.json") as f:
                mapping = json.load(f)
            hostname = mapping[hostname]["alias"]
        except Exception:
            pass

        # Find the newest file
        return http_util.Respond(
            request,
            json.dumps(
                {"value": f"<strong>{hostname}</strong>({time_diff_hrs:.1f}hrs ago)"}
            ),
            "application/json",
        )

    def _get_scalars_plugin(self):
        """Tries to get the scalars plugin.

        Returns:
        The scalars plugin or None if it is not yet registered.
        """
        return self._context.tb_context.plugin_name_to_instance.get(
            scalars_metadata.PLUGIN_NAME
        )


def _parse_request_argument(request, proto_class):
    if request.method == "POST":
        return json_format.Parse(request.data, proto_class())

    # args.get() returns the request URI-unescaped.
    request_json = request.args.get("request")
    if request_json is None:
        raise error.HParamsError(
            "Expected a JSON-formatted 'request' arg of type: %s" % proto_class
        )
    return json_format.Parse(request_json, proto_class())
