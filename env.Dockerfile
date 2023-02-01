FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN /opt/conda/bin/conda install jupyter jupyter_server>=1.11.0 scipy matplotlib tqdm scikit-learn pandas -y \
  && /opt/conda/bin/pip install tensorboardX \
  && /opt/conda/bin/pip install pyDOE \
  && /opt/conda/bin/pip install sobol_seq \
  && /opt/conda/bin/pip install torchdiffeq \
  && /opt/conda/bin/pip install fire \
  && /opt/conda/bin/pip install pickle5 \
  && /opt/conda/bin/pip install torchnet \
  && /opt/conda/bin/pip install xlrd

RUN mkdir /tmp/libs \
  && cd /tmp/libs \
  && git clone https://github.com/SsnL/PyTorch-Reparam-Module.git \
  && cd PyTorch-Reparam-Module \
  && python setup.py install


