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
Our motivation is to address the challenges in recycling by implementing a 7 DOF Robot for sorting various recyclables. Items on a conveyor belt have random positions, and the robot, equipped with a Lidar sensor, uses deep learning to identify each item's category. Open Motion Planning Library (OMPL) and Simulation Inverse Kinematics (simIK) are employed to move the object to the correct box.

## Approach
### Building Blocks:
1. Differential Kinematics (To move the robot arm joints)
2. Inverse Kinematics (To move the robot arm joints)
3. Robot Vision (To locate recyclables)
4. Neural Network for Object Detection/CNN (To identify and classify recyclables)

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
In summary, we successfully implemented a RecycleVisionBot using advanced robotics concepts. Future development could include [mention potential improvements or expansions]. The project represents a significant step toward efficient and automated recycling processes.

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


