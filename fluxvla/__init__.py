# Copyright 2026 Limx Dynamics
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

import transformers
from mmengine.utils import digit_version

transformers_minimum_version = '5.3.0'
transformers_maximum_version = '5.3.1'
transformers_version = digit_version(transformers.__version__)

assert (transformers_version >= digit_version(transformers_minimum_version) and
        transformers_version < digit_version(transformers_maximum_version)), \
    f'Transformers=={transformers.__version__} is used but incompatible. ' \
    f'Please install transformers>={transformers_minimum_version}, ' \
    f'<{transformers_maximum_version}.'

from .collators import *  # noqa: E402, F401, F403
from .datasets import *  # noqa: E402, F401, F403
from .engines import *  # noqa: E402, F401, F403
from .models import *  # noqa: E402, F401, F403
from .optimizers import *  # noqa: E402, F401, F403
from .tokenizers import *  # noqa: E402, F401, F403
from .transforms import *  # noqa: E402, F401, F403
