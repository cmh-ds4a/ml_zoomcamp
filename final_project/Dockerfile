FROM public.ecr.aws/lambda/python:3.8

RUN pip install numpy

RUN pip install sklearn

COPY lambda_function.py .

COPY model_rf.bin .

CMD [ "lambda_function.lambda_handler" ]