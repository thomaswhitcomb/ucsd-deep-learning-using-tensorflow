FROM ubuntu:latest 
RUN apt update && apt upgrade -y 
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt install -y vim
RUN apt install -y git
#
RUN git config --global user.email "thomaswhitcomb@gmail.com"
RUN git config --global user.name "Tom Whitcomb"
#
RUN pip3 install --upgrade pip
RUN echo "ee44b511e9acff615ce3a6d7b5f121a8fae247f5" > token.txt
RUN pip3 install tensorflow
RUN pip3 install Pandas
RUN pip3 install NumPy
RUN pip3 install Matplotlib
RUN pip3 install Scikit-Learn
RUN pip3 install Seaborn
RUN pip3 install SymPy

RUN python3 -c "import tensorflow as tf; print(tf.__version__)"
