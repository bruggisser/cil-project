import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from tensorflow.keras.callbacks import EarlyStopping

from model_type import ModelType

tf.get_logger().setLevel('ERROR')


class BertClassifier(ModelType):
    def __init__(self, config, logger, data_loader):
        self.modelsPath = config.get("models_path")
        self.dataLoader = data_loader
        x_train, x_val, y_train, y_val = data_loader.get_training_validation_tweets()
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        train_dataset = train_dataset.batch(32)
        val_dataset = val_dataset.batch(32)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = self.create_model()
        self.train(self.model, self.train_dataset, self.val_dataset)
        loss, acc = self.model.evaluate(self.val_dataset)
        logger.log_metric("validation_accuracy", acc)
        logger.log_metric("validation_loss", loss)

    def validate(self, tweets):
        batch_size = 32
        test_dataset = (tf.data.Dataset.from_tensor_slices(tweets).batch(batch_size))
        return tf.sigmoid(self.model.predict(test_dataset))

    def create_model(self):
        tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1'
        tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
        return tf.keras.Model(text_input, net)

    def train(self, model, x_train, x_val):
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(x_train).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-5
        opt = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps, optimizer_type="adamw")
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        callback = EarlyStopping(monitor="val_loss", patience=3)
        model.fit(x_train, epochs=epochs, validation_data=x_val, callbacks=[callback])
        model.save(self.modelsPath + self.name() + "model")

    def name(self):
        return "bert_"
