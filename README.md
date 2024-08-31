# ox-onnx


ox-onnx a clean interface lib to work with onnx models 


## intstallation :

```
pip install git+https://github.com/ox-ai/ox-onnx.git
```

## code snipptes:

```py
from ox_onnx.runtime import OnnxModel

# initialize model
# model = OnnxModel.load((model_ID="your desired onnx model")) models that have interfaced

model = OnnxModel.load(model_ID)

# desired interface model data input
data = ["""A deep learning architecture is essentially a blueprint of a neural network, outlining how data flows through multiple interconnected layers, extracting features and making decisions. Key components include input, hidden, and output layers, activation functions, weights, biases, and a loss function. Common architectures are CNNs, RNNs, LSTMs, GRUs, Transformers, GANs, and Autoencoders, each tailored for specific tasks like image recognition, natural language processing, and generative models.
"""]

# generate ouput from model
embeddings = model.generate(data)
print(embeddings)

# model Tokenization of data
en_data = model.encode(data)
de_data = model.decode(en_data)

```

## ox-onnx model interface :

for avilable model interfaces refer [model.interfaces.md](./docs/model.interfaces.md)

### currently supported models

support for other models comming soon

``` 
# Model_ID 

## interfaced :

+ "sentence-transformers/all-MiniLM-L6-v2"

```