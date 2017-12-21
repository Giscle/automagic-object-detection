# Object Detection Project Report

Team Automagic
Leader: Devin Shanahan
Members: Sonu Chauhan, Mukesh Jha

### Project Details
Our team was tasked with designing a model that could predict objects on the road, especially objects specific to roads in India.  Pretrained models did well determining Western vehicles, but failed at detecting objects unique to India roads.  We used Tensorflow and transfer learning to design our own model using annotated data from India roads.

### Pipeline
The first step was to take our video data and convert it into images, this was done using OpenCV. Next, we wrote a script to extract one out of every ten frames, the video was 30 frames per second so we had 3 frames from each second.  This provided us with 1318 images, with a good amount of variation between images.  Once the data was prepared, we moved to the annotation process.  After discussing many options, we chose the LabelMe tool for its ease of use and functionality.  Our annotations were to consist of seven categories: 1. Car, 2. Bus, 3. Truck, 4. Person, 5. Auto, 6. Motorbike, 7. Animal. LabelMe outputs the annotations in xml format, and we then needed to convert these for use with Tensorflow.  Both object detection teams worked together to annotate and shared the annotation data in a Google Cloud Storage bucket.  We split our image and xml data into a training and test set, then parsed the xml data  and created one training csv file and one test csv file.  Those csv files were then used to generate the TFRecords for Tensorflow.  With the TFRecords, we took a transfer learning approach and configured our model to train using the model checkpoints from ssd_mobilenet_v1_coco.  We then trained our model using a Google Cloud GPU and stopped around 35,000 steps due to time constrictions and only receiving minor improvements in loss at that point.  The steps for running Tensorflow on a Google Cloud VM can be found below.

https://docs.google.com/document/d/1FgyEQriEJIgDBaFYSKx-2fPsw2WhYjZzxKZOv5EMc7I/edit?usp=sharing

### Lessons
Many lessons were learned throughout this process.  The first was that data annotations need to be consistent, after checking our csv file, we realized there were some inconsistencies as well as spelling errors.  These were noticed later and had to be corrected, we should have checked the quality of the annotations immediately after generating the csv files. Another lesson came after our first training on the model yielded no results.  After analyzing our data, we realized that only one annotation per image was added to the csv.  There were over six annotations per image on average so the fix made a huge boost to the training set for the second training run. 

### Results
Overall the results were not great, but given the low amount of data they were very promising.  The model was able to detect objects such as motorbikes and autos, that were undetectable by pretrained models.  Most of the annotated data points were on cars and motorbikes and the model performed best at identifying these two objects.  Many inaccuracies occurred with objects like trucks, busses and autos because our data set included far less of those objects.  The model also failed to successfully identify objects quite often.  The model was also run on an additional dataset to see if the model generalized to new data.  This video had similar results, proving that with additional data and fine-tuning of the model, we can achieve a high level of accuracy.  The videos can be found at the link below.

https://drive.google.com/open?id=14ho9SJi1rHotM2YfVtfgeAMR1n0b9RO0

