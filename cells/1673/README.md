Neuron Model Simulation

This repository contains a single-cell neuron model simulation. The simulation can be run using Python, and it will generate and save a plot of voltage vs. time.

Setting Up the Environment

To ensure compatibility, it is recommended to create a Python virtual environment using version 3.10.8 and install the necessary libraries. However, newer versions of these libraries might also work.

Steps to Create the Virtual Environment

	1.	Create the virtual environment:

python3.10 -m venv venv


	2.	Activate the virtual environment:
	
•	On Linux/MacOS:

    source venv/bin/activate


•	On Windows:

    .\venv\Scripts\activate
    

    3.	Install the required libraries:
After activating the virtual environment, install the required libraries by running:

    pip install neuron==8.2.4 matplotlib==3.8.4

Note: While the specific versions listed above are recommended, newer versions of these libraries may also work.

Running the Simulation

To run the single-cell simulation of the neuron model, execute the following command in the same directory containing the morphology, electrophysiology and mechanisms folder:

    python neuron_simulation.py

This will run the simulation and save a plot of voltage vs. time as an image.