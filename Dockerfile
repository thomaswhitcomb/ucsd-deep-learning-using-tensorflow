FROM ubuntu:latest 
RUN apt update && apt upgrade -y 
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt install -y vim
RUN apt install -y git
RUN apt install -y sudo
#
RUN git config --global user.email "thomaswhitcomb@gmail.com"
RUN git config --global user.name "Tom Whitcomb"
#
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN pip3 install Pandas
RUN pip3 install NumPy
RUN pip3 install Matplotlib
RUN pip3 install Scikit-Learn
RUN pip3 install Seaborn
RUN pip3 install SymPy

RUN python3 -c "import tensorflow as tf; print(tf.__version__)"
