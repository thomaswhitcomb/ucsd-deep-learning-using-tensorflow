FROM ubuntu:latest 
RUN apt update
RUN apt install --assume-yes python3
RUN apt install --assume-yes python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow
RUN pip3 install Pandas
RUN pip3 install NumPy
RUN pip3 install Matplotlib
RUN pip3 install Scikit-Learn
RUN pip3 install Seaborn
RUN pip3 install SymPy

RUN python3 -c "import tensorflow as tf; print(tf.__version__)"
