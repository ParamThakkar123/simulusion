# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Default constants for LabMaze."""

import importlib

if importlib.util.find_spec("labmaze.cc.python._defaults"):
    from labmaze.cc.python._defaults import *  # pylint: disable=wildcard-import
else:
    # Fallback defaults when C++ extensions are not available.
    MAX_ROOMS = 4
    ROOM_MIN_SIZE = 3
    ROOM_MAX_SIZE = 5
    RETRY_COUNT = 1000
    EXTRA_CONNECTION_PROBABILITY = 0.0
    MAX_VARIATIONS = 26
    HAS_DOORS = False
    SIMPLIFY = True
    SPAWN_COUNT = 0
    SPAWN_TOKEN = "P"
    OBJECT_COUNT = 0
    OBJECT_TOKEN = "G"
