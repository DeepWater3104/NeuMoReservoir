from pathlib import Path
import logging
# from neuron import h
import re
import matplotlib.pyplot as plt
import subprocess
import sys
import os

def run_nrnivmodl():

    if os.path.exists("mechanisms/hippocampus/mod"):
        command = ["nrnivmodl", "mechanisms/hippocampus/mod"]
    else:
        command = ["nrnivmodl", "mechanisms"]
    
    try:
        # Use shell=True for Windows, shell=False for Unix-like systems
        shell = True if sys.platform == "win32" else False
        
        # Run the command
        result = subprocess.run(command, shell=shell, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print the output
        print(f"Ran command - {command}, \nCommand output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'nrnivmodl' command not found. Make sure NEURON is installed and in your system PATH.")


def extract_template_name(file_path):
    pattern = r'begintemplate\s+(\w+)'
    template_names = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                template_names.append(match.group(1))

    if len(template_names) > 1:
        logging.warning(f"Multiple template names found: {template_names}. Using the first one.")

    return template_names[0] if template_names else None

def get_hoc_morph_for_emodel_folder():
    """Returns the hoc morph tuple"""

    # Get the single .hoc file from the electrophysiology directory
    hoc_files = list(Path("electrophysiology").glob("*.hoc"))
    if not hoc_files:
        logging.error("No .hoc file found in 'electrophysiology'")
        raise FileNotFoundError("No .hoc file found in 'electrophysiology'")
    hoc_path = hoc_files[0]

    # Get the morphology file from the morphology directory, either .swc or .asc
    morph_files = list(Path("morphology").glob("*.swc")) + list(
        Path("morphology").glob("*.asc")
    )
    if not morph_files:
        logging.error("No morphology file (.swc or .asc) found in 'morphology'")
        raise FileNotFoundError(
            "No morphology file (.swc or .asc) found in 'morphology'"
        )
    morph_path = morph_files[0]  # take the first one if there are multiple files

    return hoc_path, morph_path

def check_line_in_file(file_path, target_line):
    with open(file_path, 'r') as file:
        for line in file:
            if target_line in line.strip():
                return True
    return False

if __name__ == "__main__":

    # Check if the mecahnisms folder exsists
    if not(os.path.exists("mechanisms")):
        logging.error("No mechanisms directory found.")

    # Run the command
    run_nrnivmodl()
    from neuron import h
    
    # Get the hoc file
    hoc_path, morph_path = get_hoc_morph_for_emodel_folder()

    #Load the standard hoc file and the custom hoc file for the model
    h.load_file('stdrun.hoc')
    h.load_file(hoc_path.as_posix())

    #Extract the template name from the hoc file, and create a cell instance
    method_name = extract_template_name(hoc_path.as_posix())

    #Based on the number of arguments in the template, initialize the cell.
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(h, method_name)(0,"morphology",morph_path.name )
    else:
        cell = getattr(h, method_name)("morphology",morph_path.name )

    #Setup the parameters for the cell
    h.celsius = 34.0
    h.v_init = -80.0

    # To see the morphology structure of the cell
    # h.topology()

    # Add a current clamp
    #Injecting a current of 0.75 nA for a duration of 1000ms with a delay of 250ms
    iclamp = h.IClamp(cell.soma[0](0.5))
    iclamp.delay = 250
    iclamp.dur = 1000
    iclamp.amp = 0.75

    # Record the membrane potential and time
    v = h.Vector().record(cell.soma[0](0.5)._ref_v)  # Membrane potential vector
    t = h.Vector().record(h._ref_t)  # Time stamp vector

    # Set the simulation time
    h.tstop = 1500

    # cvode on for faster simulation
    h.cvode_active(1)

    # Run the simulation
    h.run()

    #Plot the voltage recordings and save the figure
    plt.figure()
    plt.plot(t, v)
    plt.xlabel('Time (ms)')
    plt.ylabel('Vm (mV)')
    plt.title('Voltage plot')
    plt.show()
    plt.savefig('voltage_plot.png')
