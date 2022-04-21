open_project systolic

set_top systolic_array

add_files systolic.cpp
add_files -tb systolic_test.cpp

open_solution "solution1"
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default

csim_design -clean
csynth_design

exit