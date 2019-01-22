
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import requests
import tarfile

def decompress(path):
    t = tarfile.open(path, "w:gz")
    t.extractall(path=path.strip(".tar.gz"))

def download(url, path):
    weight_dir = '/'.join(path.split('/')[:-1])
    print(weight_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        raise RuntimeError("Downloading from url {} failed with code {}".format(url, req.status_code))
    print(path)
    with open(path, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    decompress(path)
