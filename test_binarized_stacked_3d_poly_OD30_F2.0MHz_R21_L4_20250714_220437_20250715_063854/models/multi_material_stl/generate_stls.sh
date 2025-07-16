#!/bin/bash
# Generate all STL files from OpenSCAD

echo 'Generating lens 1 Agilus30...'
openscad -o lens_1_Agilus30_RGD_FFF00064.stl lens_1_Agilus30.scad
echo 'Generating lens 1 VeroClear...'
openscad -o lens_1_VeroClear_RGD_810.stl lens_1_VeroClear.scad

echo 'Generating lens 2 Agilus30...'
openscad -o lens_2_Agilus30_RGD_FFF00064.stl lens_2_Agilus30.scad
echo 'Generating lens 2 VeroClear...'
openscad -o lens_2_VeroClear_RGD_810.stl lens_2_VeroClear.scad

echo 'Generating lens 3 Agilus30...'
openscad -o lens_3_Agilus30_RGD_FFF00064.stl lens_3_Agilus30.scad
echo 'Generating lens 3 VeroClear...'
openscad -o lens_3_VeroClear_RGD_810.stl lens_3_VeroClear.scad

echo 'Generating lens 4 Agilus30...'
openscad -o lens_4_Agilus30_RGD_FFF00064.stl lens_4_Agilus30.scad
echo 'Generating lens 4 VeroClear...'
openscad -o lens_4_VeroClear_RGD_810.stl lens_4_VeroClear.scad

