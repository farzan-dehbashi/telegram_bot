from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

with open ('token.txt', 'r') as f:
    TOKEN = f.read()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255


class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

def start (update, context):
    update.message.reply_text('Welcome! Here is Farzan Bot')

def help (update, context):
    update.message.reply_text('''
    /start -> starts a conversation
    /help -> help
    /train -> trains
    ''')

def train (update, context):
    update.message.reply_text('Model is being trained...')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
    model.fit(x_train, y_train, epochs= 10, validation_data=(x_test, y_test))
    model.save('cifar_classifier.model')
    update.message.reply_text('Training is done')


def handle_message(update, context):
    update.message.reply_text(' please train the  model and send a pic')

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype = np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)

    prediction = model.predict(np.array([img / 255 ]))
    update.message.reply_text(f'In this image Farzan bot sees {class_names[np.argmax(prediction)]}')


updater = Updater(TOKEN, use_context= True)
dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))
dp.add_handler(CommandHandler('help', help))
dp.add_handler(CommandHandler('train', train))

dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()