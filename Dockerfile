FROM ubuntu:latest 
RUN apt update && apt upgrade -y 
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y sudo
RUN apt-get install net-tools
RUN apt install -y curl git-core gcc make zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev
RUN apt-get install -y tmux
#
RUN git config --global user.email "thomaswhitcomb@gmail.com"
RUN git config --global user.name "Tom Whitcomb"
RUN cd $HOME
RUN git clone https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN pyenv install anaconda3-5.3.0
#
RUN pip3 install --upgrade pip
RUN pip3 install conda
RUN pip3 install pylint
RUN pip3 install tensorflow
RUN pip3 install Pandas
RUN pip3 install NumPy
RUN pip3 install Matplotlib
RUN pip3 install Scikit-Learn
RUN pip3 install Seaborn
RUN pip3 install SymPy
RUN pip3 install pillow
RUN pip3 install keras

COPY bashrc /root/.bashrc
COPY tmux /root/.tmux.conf
COPY git-completion.bash /root
RUN python3 -c "import tensorflow as tf; print(tf.__version__)"
