from ox_onnx.runtime import OnnxModel


data = """
A deep learning architecture is essentially a blueprint of a neural network, outlining how data flows through multiple interconnected layers, extracting features and making decisions. Key components include input, hidden, and output layers, activation functions, weights, biases, and a loss function. Common architectures are CNNs, RNNs, LSTMs, GRUs, Transformers, GANs, and Autoencoders, each tailored for specific tasks like image recognition, natural language processing, and generative models.
"""*10
model = OnnxModel().load()
embeddings = model.generate([data])

print(embeddings)


en_data = model.encode(data)
de_data = model.decode(en_data)