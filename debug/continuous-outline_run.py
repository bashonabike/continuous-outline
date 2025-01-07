#!C:\Program Files\Inkscape\bin\pythonw.exe
# 2025-01-07 14:31:39.742836
import sys
import os
sys.path.append('C:\Users\Dell 5290\AppData\Roaming\inkscape\extensions\continuous-outline')
sys.path.append('C:\Users\Dell 5290\AppData\Roaming\inkscape\extensions')
sys.path.append('C:\Program Files\Inkscape\share\inkscape\extensions')
sys.path.append('C:\Program Files\Inkscape\share\inkscape\extensions\inkex\deprecated-simple')
sys.path.append('C:\Program Files\Inkscape\lib\python311.zip')
sys.path.append('C:\Program Files\Inkscape\lib\python3.11')
sys.path.append('C:\Program Files\Inkscape\lib\python3.11\lib-dynload')
sys.path.append('C:\Program Files\Inkscape\lib\python3.11\site-packages')
sys.path.append('C:\Program Files\Inkscape\share\inkscape\extensions\inkex\deprecated-simple')
args = [
    "--id=path4",
    "--id=rect4",
    "--id=image1",
    "--id=rect3",
    "--id=rect2",
    "--id=rect1",
    "--tab=tab_general",
    "--keeporiginal=false",
    "--ltres=1.0",
    "--qtres=1.0",
    "--pathomit=8",
    "--rightangleenhance=true",
    "--colorsampling=0",
    "--numberofcolors=16",
    "--mincolorratio=0",
    "--colorquantcycles=3",
    "--layering=0",
    "--strokewidth=1.0",
    "--linefilter=false",
    "--roundcoords=1",
    "--viewbox=false",
    "--desc=false",
    "--blurradius=1",
    "--blurdelta=20.0",
    "C:\Users\Dell 5290\AppData\Roaming\inkscape\extensions\continuous-outline\debug\input_file.svg"
]

# Run the extension.
import continuous-outline as extension
# Find the inkex.Effect class
for item in dir(extension):
    # Brute force looking for 'run' attribute.
    if hasattr(getattr(extension, item), "run"):
        # Create an instance of the extension.
        extension_instance = getattr(extension, item)()
        # Run the extension with the given arguments.
        output = os.path.join("C:\Users\Dell 5290\AppData\Roaming\inkscape\extensions\continuous-outline\debug", "output_file.svg")
        extension_instance.run(
            args = args,
            output = output)
        break
