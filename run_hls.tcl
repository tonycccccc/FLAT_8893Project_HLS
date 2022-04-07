# Create a project
open_project -reset flat

# Set the top-level function
set_top FlatDataflow

# Add design files
add_files flat.h
add_files utils.cpp
add_files convolution.cpp 
# Add test bench & files
add_files -tb flat_test.cpp
# Add memory files
add_files -tb ./query_file.bin

# ########################################################
# Create a solution
open_solution -reset solution1
# Define technology and clock rate
set_part {xc7z020-clg400-1}
create_clock -period 10

# Source x_hls.tcl to determine which steps to execute
source x_hls.tcl
csim_design
# Set any optimization directives
# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design
	
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	
	cosim_design
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	
	cosim_design
	export_design
} else {
	# Default is to exit after setup
	csynth_design
}

exit
