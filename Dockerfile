FROM tensorflow/tensorflow:nightly-gpu
MAINTAINER yjc133

RUN apt-get -y update
RUN apt-get install -y tmux
RUN apt-get install -y libsndfile1
RUN apt-get install -y python3.7 
RUN pip3 install soundfile

CMD ["bash"]
