import os
import warnings 
from pyopencl import CompilerWarning
import time
import pyopencl as cl
import numpy as np

# Suppress OpenCL compiler warnings
warnings.simplefilter("ignore", CompilerWarning)

# Suppress OpenCL compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


# Load the OpenCL context and queue
platform = cl.get_platforms()[0]  # Select the first platform
device = platform.get_devices()[0]  # Select the first device
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Read the OpenCL kernel file
with open("state_machine.cl", "r") as kernel_file:
    kernel_code = kernel_file.read()

# Build the kernel program
program = cl.Program(context, kernel_code).build()

# Kernel variables
state = np.zeros(8, dtype=np.uint32)  # 256-bit state (8 x 32-bit words)
entropy_input = np.random.randint(0, 256, size=64, dtype=np.uint8)  # Random entropy input
output = np.zeros(32, dtype=np.uint8)  # Buffer to hold generated random data

# Generate time seed based on current time (could use nanoseconds for more precision)
time_seed = np.array([int(time.time_ns())], dtype=np.uint64)

# Create buffers
state_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=state)
entropy_input_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=entropy_input)
output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)
time_seed_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=time_seed)

# Set kernel arguments
kernel = program.crng_kernel
kernel.set_args(state_buf, output_buf, entropy_input_buf, time_seed_buf, np.int32(entropy_input.size))

# Execute the kernel
global_size = (1,)  # One thread
cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)

# Read the results from the output buffer
cl.enqueue_copy(queue, output, output_buf).wait()

# Convert the output to hexadecimal and print it
hex_output = ''.join(f'{byte:02x}' for byte in output)
print("Generated random data (hex):", hex_output)
