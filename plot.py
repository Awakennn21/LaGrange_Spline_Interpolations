import sys
import os
import matplotlib.pyplot as plt

def read_data(filename):
    x_data = []
    y_data = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = line.strip().split()
            x_data.append(float(x))
            y_data.append(float(y))
    return x_data, y_data

if len(sys.argv) != 4:
    print("Please provide two file paths and the output file name as command-line arguments.")
    sys.exit(1)

filename1 = sys.argv[1]
filename2 = sys.argv[2]
output_filename = sys.argv[3]

chelm_x, chelm_y = read_data(filename1)
other_x, other_y = read_data(filename2)

plt.plot(chelm_x, chelm_y, 'o', markersize=2, label='Data')  # Plot as scattered points
plt.plot(other_x, other_y, label='Interpolated')
plt.xlabel('Distance[m]')
plt.ylabel('Height[m]')
plt.yscale('log')
plt.title(output_filename)

# Create the "Plots" directory if it doesn't exist
if not os.path.exists("Plots"):
    os.makedirs("Plots")

# Save the plot as a PNG file in the "Plots" directory
output_path = os.path.join("Plots", output_filename)
plt.savefig(output_path)

plt.legend()
plt.show()