# use python image as a base
FROM python:3.7.4-buster
# set a working directory (inside the container, not host os)
WORKDIR /usr/src/app


# install dependencies
RUN apt-get update -y && apt-get upgrade -y
# add postgresql-dev gcc python3-dev musl-dev freetds-dev g++
RUN pip install --upgrade pip

COPY . .

# install dependencies 
RUN pip install -r requirements.txt

# execute the app when image will start 
CMD [ "echo", "Starting docker" ]
CMD [ "python", "./logistic_regression.py" ]
