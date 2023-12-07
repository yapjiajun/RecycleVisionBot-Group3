<h1 align="center">
    <a>
    <img src="./.github/assets/logo.png">
    </a>
</h1>

<h1 align="center"> RecycleVisionBot (Group 3) </h1>

<p align="center">
  <i align="center">A Recycle Robot that uses ML/AI to sort through recyclables ♻️ </i>
</p>

## Introduction
Given the current recycling rate of 32% in the United States (EPA, 2018), it is evident that refining our recyclable sorting methods is imperative to avoid misclassification and landfill disposal. This prompted the exploration of a robotic sorting system capable of accurately and efficiently sorting various recyclables on a conveyor belt. The objective is to overcome misclassification challenges, enhance overall recycling efficiency, and apply the knowledge acquired from the ECSE275 course to advance this project.

## Approach
### Building Blocks (Concepts):
#### Inverse Kinematics:
Inverse Kinematics (IK) is a way for robots to calculate how to move their joints to reach a particular position or angle. 

Utilizing the principles of Inverse Kinematic (IK), our approach allows the Franka Emika Panda 7 DOF robot to move its robotic arm joints with precision and efficiency by using the IK Solver Function, the system calculates the joint configurations necessary to achieve a desired end-effector position, ensuring optimal coordination and dexterity in sorting recyclables with utmost accuracy.

#### Robot Vision:
Robot Vision involves equipping robots with the ability to perceive and interpret visual information from their environment. It enables robots to "see" and make decisions based on the visual input they receive.

To enable the robot to precisely locate recyclables on the conveyor belt, we implement advanced robotic vision concepts. This involves tasks such as camera calibration, greyscaling the image, and coordinating transformations to accurately pinpoint the actual location of recyclables from the camera's perspective. By incorporating these vision techniques, we are able to allow the system to interpret visual data and make informed decisions in real-time, contributing to the efficiency and accuracy of the recycling process.

#### Neural Network for Object Detection/CNN:
Neural Network for Object Detection is a smart computer system that can spot and categorize things in pictures or videos. Convolution Neural Network (CNN), a specific type of neural network, are great at understanding visual details, making them useful for tasks like recognizing objects. 

To ensure our system accurately classifies recyclables, we utilized MOSTAFA MOHAMED's Garbage Classification Kaggle Dataset. We chose out of the total 12 types, we picked 6 for our application: cardboard, glass, metal, paper, plastic, and general trash. Using a CNN Model, we trained our datasets for our specific use case, completing the training over 100 epochs.

<h2>
    <a>
    <img src="./.github/assets/newplot.png">
    </a>
</h2>

Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data

### Implementation:
- [Brief explanation of implementation choices]

### Flow Chart:
[Insert flow chart here]

### Data Flow:
[Describe how different components interface with each other]

## Results
- Images of final implementation
- GIFs showcasing system in action
- Embedded videos from YouTube
- Quantitative data presented in tables or plots
- Qualitative performance discussion
- Evaluation against pre-determined success metrics

## Conclusion
In summary, the successful implementation of the RecycleVisionBot (ReViBot) showcases the practical application of concepts acquired in our coursework. By seamlessly integrating machine learning models and computer vision, the system accurately categorizes a diverse array of recyclables, directing the robotic arm to execute precise sorting into designated compartments. Looking forward, our future developments include the exploration of alternative models such as the YOLOv8 Architecture, and the incorporation of datasets from various sources to enhance the model's adaptability to a broader spectrum of recyclable items and environmental conditions.

## Contributors
`Jules Siegel (Leader)`: Spearheading the integration of the environment in CoppeliaSim by skillfully incorporating the Franka Emika Panda 7 DOF Robotic Arm and seamlessly integrating a conveyor belt system. Responsible for meticulously configuring the initial environment settings to ensure optimal functionality.

`Emily Hawkins (Test)`: Conducting rigorous Robot Performance Testing to assess the efficacy of various Machine Learning Models in efficiently sorting recyclables. Utilizing a comprehensive approach to evaluate and identify the best-performing models that meet the project's requirements.

`Jacob Boling (ML/AI)`: Taking charge of the training process for diverse Machine Learning Models, experimenting with different parameters to enhance their performance. Engaging in thorough exploration and optimization to achieve superior results in model training.

`Yap Jia Jun (Document)`: Playing a pivotal role in the project's documentation phase by capturing and articulating detailed findings and results. Creating comprehensive documentation that not only highlights key insights but also serves as a valuable resource for future reference.

`Juan Beaver (Integration)`: Orchestrating the seamless integration of trained Machine Learning Models into CoppeliaSim, ensuring a cohesive and efficient implementation. Focusing on the smooth transition from training to real-world application, Juan plays a crucial role in bridging the gap between theoretical models and practical simulation environments.

[//]: contributor-faces
<a href="https://github.com/JulesSiegel"><img src="https://avatars.githubusercontent.com/u/152318869?v=4" title="jules-siegel" width="50" height="50"></a>
<a href="https://github.com/emilyghawk"><img src="https://avatars.githubusercontent.com/u/152319430?v=4" title="emily-g-hawk" width="50" height="50"></a>
<a href="https://github.com/JacobBoling"><img src="https://avatars.githubusercontent.com/u/82610978?v=4" title="jacob-boling" width="50" height="50"></a>
<a href="https://github.com/yapjiajun"><img src="https://avatars.githubusercontent.com/u/79196462?v=4" title="yap-jia-jun" width="50" height="50"></a>
<a href="https://github.com/juanbeaver"><img src="https://avatars.githubusercontent.com/u/27016289?v=4" title="juan-beaver" width="50" height="50"></a>

## License
[MIT](https://choosealicense.com/licenses/mit/)


