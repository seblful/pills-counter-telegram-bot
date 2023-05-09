#!/usr/bin/python

import telebot
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from io import BytesIO
from shapely.geometry import Polygon

COLORS = [(98, 231, 4), (228, 161, 0)] # Green and blue
CLASSES = ['capsules', 'tablets']

API_TOKEN = '5727108383:AAG-oPn1UgB39Z_KbKqAyn8hRSjQETT60XM'

bot = telebot.TeleBot(API_TOKEN)

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Hi there, I am Pills Counter bot.

I am here to help you with counting pills, \
just send me photo and I will send you another \
and then I will say how many tablets or capsules \
that picture has.\
""")
    

# Handle images
@bot.message_handler(content_types=['photo'])
def photo(message):
    # Get file ID of the photo sent by the user
    file_id = message.photo[-1].file_id

    # Download the photo file from Telegram servers
    file_info = bot.get_file(file_id)
    file = BytesIO(bot.download_file(file_info.file_path))

    # Load the image using OpenCV
    image = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    # Prediction
    predicted_image, count_dict = get_prediction(image)

    # Send a confirmation message
    message_to_send = f"""There are {count_dict.get('capsules', 0)} capsules and {count_dict.get('tablets', 0)} tablets. \
A total of {count_dict.get('capsules', 0) + count_dict.get('tablets', 0)} pills."""
    bot.reply_to(message, message_to_send)

    # Encode the filtered image as a JPEG and send it back to the user
    ret, buffer = cv2.imencode('.jpg', predicted_image)
    file = BytesIO(buffer)
    bot.send_photo(message.chat.id, file, caption="There is your predicted image.")


# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    message_to_send = f"I'm just Pills Counter bot and i can't reply to your message. Send me pictures with pills, please."
    bot.reply_to(message, message_to_send)


def get_prediction(image):
    '''
    Gets image from telebot, make predictions, 
    counts predicted classes, draws dots on image,
    returns dict with counts and labelled image
    '''
    # Load a model
    model = YOLO('yolov8l.pt')  # load an official model
    model = YOLO('best.pt')  # load a custom model

    # Get prediction
    prediction = model(image)

    # Get predicted classes
    predicted_classes = prediction[0].boxes.cls
    # Get predicted confidence of each class
    prediction_confidences = prediction[0].boxes.conf

    # Get polygons
    polygons =  prediction[0].masks.xy
    # Convert polygons to int32
    polygons = [polygon.astype(np.int32) for polygon in polygons]

    # Create indices mask that shows what is overlapping polygon has smaller confidence score
    indices_mask = remove_overlapping_polygons(polygons, prediction_confidences)

    # Create new fixed lists with predicted classes and polygons
    fixed_predicted_classes = predicted_classes[np.array(indices_mask, dtype=bool)]
    fixed_polygons = [polygons[i] for i in range(len(indices_mask)) if indices_mask[i] == 1]
    # fixed_predicted_classes = [predicted_classes[i] for i in range(len(indices_mask)) if indices_mask[i] == 1]

    # Get counts of classes
    unique, counts = torch.unique(fixed_predicted_classes, return_counts=True)
    # Get dicts with counts of classes
    count_dict = {CLASSES[int(key)]: value for key, value in zip(unique.tolist(), counts.tolist())}

    # # Draw polygons
    # for polygon, predicted_class in zip(fixed_polygons, fixed_predicted_classes):
    #     cv2.polylines(image, [polygon], True, COLORS[int(predicted_class)])

    # Draw dots
    for polygon, predicted_class in zip(fixed_polygons, fixed_predicted_classes):
         # Find center of polygon
         center_coordinates =  (np.mean(polygon[:, 0], dtype=np.int32), np.mean(polygon[:, 1], dtype=np.int32)) # x and y respectively
         # Draw a circle
         cv2.circle(image, center_coordinates, 5, COLORS[int(predicted_class)], 2, cv2.LINE_AA)

    # # Show image with predictions on it
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image, count_dict


def remove_overlapping_polygons(polygons, prediction_confidences):
    '''
    Takes polygons, finds overlapping regions,
    intersection area, overlap percentage,
    creates indices mask that shows what 
    overlapping polygon has smaller confidence.
    '''

    # Convert the NumPy arrays to Shapely polygons
    shapely_polygons = [Polygon(polygon) for polygon in polygons]
    # Create an empty list with overlapping pairs
    overlapping_pairs = []
    
    # Check for overlaps between all pairs of polygons
    for i in range(len(shapely_polygons)):
        for j in range(i+1, len(shapely_polygons)):
            if shapely_polygons[i].intersects(shapely_polygons[j]):
                # Calculate the percentage of overlap
                intersection_area = shapely_polygons[i].intersection(shapely_polygons[j]).area
                overlap_percentage = intersection_area / shapely_polygons[i].area
                # Add overlapping polygons indexes to list
                if overlap_percentage > 0.5:
                    overlapping_pairs.append((i, j))

    # Mask of remains indices
    indices_mask = [1 for i in range(len(shapely_polygons))]

    # Remove one of the overlapping polygons
    for first_over_polygon_ind, second_over_polygon_ind in overlapping_pairs:
        # Find index that has the smallest prediction confidence
        first_has_bigger_conf = prediction_confidences[first_over_polygon_ind] >= prediction_confidences[second_over_polygon_ind]
        index_small_conf = [first_over_polygon_ind, second_over_polygon_ind][first_has_bigger_conf]
        # Set value with smaller confidence to 0 in indices_mask
        indices_mask[index_small_conf] = 0
    
    return indices_mask


bot.infinity_polling()