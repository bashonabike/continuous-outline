import inkex
import os
import urllib.parse as urllib_parse
import urllib.request as urllib_request
from PIL import Image, ImageDraw
from io import BytesIO
from lxml import etree
import base64
import helpers.edge_detect.edge_detection as edge
import copy as cp
import numpy as np

"""
Extension for InkScape 1.X
Features
 - Trace elegant artful continuous line drawing around your image, with highlights as specified, for nice contouring border
 
Author: Liam Cline
Mail: liamcline@gmail.com

"""

#TODO: Input all svg paths elipses, rectangles, etc. pick all pixels inside bounds as areas to emphasize detail

class Continuous_outline(inkex.EffectExtension):

    def checkImagePath(self, element):
        """
        Check if the image path is valid.

        Parameters
        ----------
        element : inkex.Element
            The element to check.

        Returns
        -------
        str
            The valid path of the image.
        """
        xlink = element.get('xlink:href')
        if xlink and xlink[:5] == 'data:':
            # No need, data already embedded
            return

        url = urllib_parse.urlparse(xlink)
        href = urllib_request.url2pathname(url.path)

        # Primary location always the filename itself.
        path = self.absolute_href(href or '')

        # Backup directory where we can find the image
        if not os.path.isfile(path):
            path = element.get('sodipodi:absref', path)

        if not os.path.isfile(path):
            inkex.errormsg('File not found "{}". Unable to embed image.').format(path)
            return

        if (os.path.isfile(path)):
            return path

    def add_arguments(self, pars):
        """
        Add the command line arguments.

        Parameters
        ----------
        pars : argparse.ArgumentParser
            The parser to add arguments to.

        """
        pars.add_argument("--tab")
        pars.add_argument("--keeporiginal", type=inkex.Boolean, default=False, help="Keep original image on canvas")
        pars.add_argument("--ltres", type=float, default=1.0, help="Error treshold straight lines")
        pars.add_argument("--qtres", type=float, default=1.0, help="Error treshold quadratic splines")
        pars.add_argument("--pathomit", type=int, default=8, help="Noise reduction - discard edge node paths shorter than")         
        pars.add_argument("--rightangleenhance", type=inkex.Boolean, default=True, help="Enhance right angle corners")
        pars.add_argument("--colorsampling", default="2",help="Color sampling")      
        pars.add_argument("--numberofcolors", type=int, default=16, help="Number of colors to use on palette")
        pars.add_argument("--mincolorratio", type=int, default=0, help="Color randomization ratio")
        pars.add_argument("--colorquantcycles", type=int, default=3, help="Color quantization will be repeated this many times")           
        pars.add_argument("--layering", default="0",help="Layering")
        pars.add_argument("--strokewidth", type=float, default=1.0, help="SVG stroke-width")
        pars.add_argument("--linefilter", type=inkex.Boolean, default=False, help="Noise reduction line filter")
        #pars.add_argument("--scale", type=float, default=1.0, help="Coordinate scale factor")
        pars.add_argument("--roundcoords", type=int, default=1, help="Decimal places for rounding")
        pars.add_argument("--viewbox", type=inkex.Boolean, default=False, help="Enable or disable SVG viewBox")
        pars.add_argument("--desc", type=inkex.Boolean, default=False, help="SVG descriptions")
        pars.add_argument("--blurradius", type=int, default=0, help="Selective Gaussian blur preprocessing")
        pars.add_argument("--blurdelta", type=float, default=20.0, help="RGBA delta treshold for selective Gaussian blur preprocessing")
  

    def image_prep(self, image):
        """
        Prepare an image for use in the continuous outline algorithm.

        Parameters
        ----------
        image : inkex.Element
            The element containing the image to be prepared.

        Returns
        -------
        str
            The path to the prepared image.

        This function checks if an image is embedded or linked, and if so, it extracts the image from the SVG element.
        It then writes the image to a temporary directory, and returns the path to the image.

        """
        self.path = self.checkImagePath(image)  # This also ensures the file exists
        if self.path is None:  # check if image is embedded or linked
            image_string = image.get('{http://www.w3.org/1999/xlink}href')
            # find comma position
            i = 0
            while i < 40:
                if image_string[i] == ',':
                    break
                i = i + 1
            img = Image.open(BytesIO(base64.b64decode(image_string[i + 1:len(image_string)])))
        else:
            img = Image.open(self.path)

        # Write the embedded or linked image to temporary directory
        if os.name == "nt":
            exportfile = "helpers.png"
        else:
            exportfile = "/tmp/helpers.png"

        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(exportfile, "png")

        return exportfile

    def get_image_offsets(self, svg_image):
        # Determine image offsets
        """
        Determine image offsets.

        Parameters
        ----------
        svg_image : inkex.Element
            The element containing the image to get offsets from.

        Returns
        -------
        tuple
            A tuple containing the x and y offsets of the image.

        This function determines the offset of the image from the origin of the SVG.
        It does this by traversing the SVG tree upwards from the image element, and summing
        the x and y coordinates of each element. The result is the offset of the image
        from the origin of the SVG.
        """
        parent = svg_image.getparent()
        if parent is not None and parent != self.document.getroot():
            tpc = parent.composed_transform()
            x_offset = tpc.e + float(svg_image.get('x'))
            y_offset = tpc.f + float(svg_image.get('y'))
        else:
            x_offset = float(svg_image.get('x'))
            y_offset = float(svg_image.get('y'))

        return (x_offset, y_offset)

    def fit_svg(self, exportfile, image):
        # Delete the temporary png file again because we do not need it anymore
        """
        Fit the SVG image back into the SVG document tree.

        Parameters
        ----------
        exportfile : str
            The path to the temporary SVG file containing the image.
        image : inkex.Element
            The element containing the image to be fitted back into the SVG document tree.

        Returns
        -------
        inkex.Group
            The element containing the fitted image.

        This function deletes the temporary PNG file and parses the temporary SVG file.
        It then inserts the parsed SVG as a new group into the current document tree.
        The group is then transformed to fit the original XY coordinates and width/height of the image.
        Finally, the function deletes the temporary SVG file and returns the new group element.
        """
        if os.path.exists(exportfile):
            os.remove(exportfile)

        # new parse the SVG file and insert it as new group into the current document tree
        doc = etree.parse(exportfile + ".svg").getroot()
        newGroup = self.document.getroot().add(inkex.Group())
        trace_width = None
        trace_height = None
        if doc.get('width') is not None and doc.get('height') is not None:
            trace_width = doc.get('width')
            trace_height = doc.get('height')
        else:
            viewBox = doc.get('viewBox')  # eg "0 0 700 600"
            trace_width = viewBox.split(' ')[2]
            trace_height = viewBox.split(' ')[3]

        # add transformation to fit previous XY coordinates and width/height
        # image might also be influenced by other transformations from parent:
        parent = image.getparent()
        if parent is not None and parent != self.document.getroot():
            tpc = parent.composed_transform()
            x_offset = tpc.e
            y_offset = tpc.f
        else:
            x_offset = 0.0
            y_offset = 0.0
        img_w = image.get('width')
        img_h = image.get('height')
        img_x = image.get('x')
        img_y = image.get('y')
        if img_w is not None and img_h is not None and img_x is not None and img_y is not None:
            # if width/height are not unitless but end with px, mm, in etc. we have to convert to a float number
            if img_w[-1].isdigit() is False:
                img_w = self.svg.uutounit(img_w)
            if img_h[-1].isdigit() is False:
                img_h = self.svg.uutounit(img_h)

            transform = "matrix({:1.6f}, 0, 0, {:1.6f}, {:1.6f}, {:1.6f})" \
                .format(float(img_w) / float(trace_width), float(img_h) / float(trace_height), float(img_x) + x_offset,
                        float(img_y) + y_offset)
            newGroup.attrib['transform'] = transform
        else:
            t = image.composed_transform()
            img_w = t.a
            img_h = t.d
            img_x = t.e
            img_y = t.f
            transform = "matrix({:1.6f}, 0, 0, {:1.6f}, {:1.6f}, {:1.6f})" \
                .format(float(img_w) / float(trace_width), float(img_h) / float(trace_height), float(img_x) + x_offset,
                        float(img_y) + y_offset)
            newGroup.attrib['transform'] = transform

        for child in doc.getchildren():
            newGroup.append(child)

        # Delete the temporary svg file
        if os.path.exists(exportfile + ".svg"):
            os.remove(exportfile + ".svg")

        return newGroup

    def isolate_sub_images(self, detail_bounds, export_file, image_in_svg):
        #Determine image offsets
        """
        Isolate sub-images from a larger image.

        Parameters
        ----------
        detail_bounds : list of inkex.Elements
            A list of elements containing the bounding boxes of the sub-images to be isolated.
        export_file : str
            The path to the larger image file containing the sub-images.
        image_in_svg : inkex.Element
            The element containing the larger image to be isolated from.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing the local origin of the sub-image in the larger image, the image object itself, and the path to the saved sub-image file.

        This function takes a list of bounding boxes and an image file as input, and outputs a list of dictionaries, each containing the local origin of the sub-image in the larger image, the image object itself, and the path to the saved sub-image file.
        The function first determines the offset of the larger image in the SVG document tree, and then scales the bounding box coordinates to match the size of the larger image.
        It then crops the bounding box of each sub-image from the larger image, and saves the resulting image to a temporary file.
        Finally, the function returns a list of dictionaries, each containing the local origin of the sub-image in the larger image, the image object itself, and the path to the saved sub-image file.
        """
        (x_offset, y_offset) = self.get_image_offsets(image_in_svg)

        with Image.open(export_file) as image:
            image_size = image.size

            x_scale = image_size[0]/float(image_in_svg.get('width'))
            y_scale = image_size[1]/float(image_in_svg.get('height'))

            sub_image_counter = 0
            bounds_dicts = []

            for bounds in detail_bounds:

                # Check if ellipse or rect
                if bounds.tag == inkex.addNS('rect', 'svg'):
                    # Get rectangle properties
                    #TODO: figure out way to crop by float
                    x = int((float(bounds.attrib.get('x', 0)) - x_offset)*x_scale)
                    y = int((float(bounds.attrib.get('y', 0)) - y_offset)*y_scale)
                    width = int(float(bounds.attrib.get('width', 0))*x_scale)
                    height = int(float(bounds.attrib.get('height', 0))*y_scale)
                    local_origin = (x, y)

                    # Crop the image to the rectangle
                    bbox = (x, y, x + width, y + height)

                    cropped_image = image

                elif bounds.tag == inkex.addNS('ellipse', 'svg'):
                    # Get ellipse properties
                    cx = int((float(bounds.attrib.get('cx', 0)) - x_offset)*x_scale)
                    cy = int((float(bounds.attrib.get('cy', 0)) - y_offset)*y_scale)
                    rx = int(float(bounds.attrib.get('rx', 0))*x_scale)
                    ry = int(float(bounds.attrib.get('ry', 0))*y_scale)
                    local_origin = np.array([cx - rx, cy - ry])

                    # Create a mask for the ellipse
                    mask = Image.new('L', image.size, 0)
                    draw = ImageDraw.Draw(mask)
                    bbox = (cx - rx, cy - ry,
                            cx + rx, cy + ry)
                    draw.ellipse(bbox, fill=255)

                    # Apply the mask to the image
                    cropped_image = Image.new('RGBA', image.size)
                    self.msg(str(image.size[0]))
                    cropped_image.paste(image, (0, 0), mask)
                else:
                    raise inkex.AbortExtension("Only ellipses and rectangles are supported as bounds.")

                # Crop the bounding box of the rect or ellipse
                cropped_image = cropped_image.crop(bbox)

                # Save the resulting image
                #TODO: need to save temp sub images?
                output_path = "sub_image_" + str(sub_image_counter) + ".png"
                sub_image_counter += 1
                cropped_image.save(output_path)
                inkex.utils.debug(f"Saved cropped image to {output_path}")

                #Save to dict
                bounds_dicts.append({"localorigin": local_origin, "imageobject": cropped_image,
                    "imagepath": output_path})

        return bounds_dicts

    def effect(self):
        """
        Internal overwrite for scale: overwrite scale to 1.0

        Effect for continuous outline algorithm

        Parameters
        ----------
        self : object
            The effect object.

        Returns
        -------
        None

        Notes
        -----
        This function is the main entry point for the effect. It is called by Inkscape when
        the effect is invoked by the user.

        The function first checks if the selection is empty. If so, it displays a message to the user.
        If the selection is not empty, it then checks if the selection contains any images. If not, it displays a message to the user.
        If the selection does contain images, it then processes each image in the selection.

        The function extracts the image from the SVG element, resizes it to the desired size, and then runs the image through the edge detection
        algorithm. The resulting image is then saved to a temporary file.

        The function then isolates sub-images from the main image by cropping the image to the bounding box of each selected detail.

        The function then applies the edge detection algorithm to each sub-image and saves the resulting images to temporary files.

        The function then builds the path commands for the resulting images.

        Finally, the function creates the inkex.Path element and adds it to the SVG document tree.

        """
        # internal overwrite for scale:
        self.options.scale = 1.0
    
        if len(self.svg.selected) > 0:
            images = self.svg.selection.filter(inkex.Image).values()
            detail_bounds= self.svg.selection.filter(inkex.Rectangle, inkex.Ellipse).values()

            if len(images) > 0:
                for svg_image in images:
                    exportfile = self.image_prep(svg_image)
                    detail_sub_dicts = self.isolate_sub_images(detail_bounds, exportfile, svg_image)
                    edge.k_means_clustering(exportfile)
                    main_image_outline = edge.detect_edges('clustered.png')
                    contours_all = list(cp.deepcopy(main_image_outline))
                    # edge.vectorize_edgified_image(contours_all)

                    with Image.open(exportfile) as image:
                        image_size = image.size

                        x_scale = image_size[0] / float(svg_image.get('width'))
                        y_scale = image_size[1] / float(svg_image.get('height'))
                        scale_nd = np.array([x_scale, y_scale])

                    # Determine image offsets
                    main_image_offsets = np.array(self.get_image_offsets(svg_image))

                    #Offset main contours to line up with master photo on svg
                    contours_transformed = []
                    for contour in contours_all:
                        new_contour = []
                        for point in contour:
                            new_contour.append(list((point/scale_nd + main_image_offsets)[0]))
                        contours_transformed.append(new_contour)


                    for detail_sub_dict in detail_sub_dicts:
                        path = detail_sub_dict["imagepath"]
                        detail_outline = edge.detect_edges(path)

                        #Offset detail to line up with master photo
                        for contour in detail_outline:
                            new_contour = []
                            for point in contour:
                                new_contour.append(list(((point + detail_sub_dict["localorigin"])/scale_nd +
                                                         main_image_offsets)[0]))
                            contours_transformed.append(new_contour)

                    #TODO: need to save temp sub images?
                    # Build the path commands
                    commands = []

                    for contour in contours_transformed:
                        for i, point in enumerate(contour):
                            if i == 0:
                                commands.append(['M', point])  # Move to the first point
                            else:
                                commands.append(['L', point])  # Line to the next point
                            # self.msg(str(point))
                        # commands.append(['Z'])  # Close path

                    # Create the inkex.Path
                    path = inkex.paths.Path(commands)

                    # Add a new path element to the SVG
                    path_element = inkex.PathElement()
                    path_element.style = {'stroke': 'black', 'fill': 'none'}
                    path_element.set('d', str(path))  # Set the path data
                    self.svg.get_current_layer().append(path_element)
                    #TODO: units are in pixels, scaling it's way too big why

                    # nodeclipath = os.path.join("imagetracerjs-master", "nodecli", "nodecli.js")
                    #
                    # ## Build up imagetracerjs command according to your settings from extension GUI
                    # command = "node --trace-deprecation " # "node.exe" or "node" on Windows or just "node" on Linux
                    # if os.name=="nt": # your OS is Windows. We handle path separator as "\\" instead of unix-like "/"
                    #     command += str(nodeclipath).replace("\\", "\\\\")
                    # else:
                    #     command += str(nodeclipath)
                    # command += " " + exportfile
                    # command += " ltres "             + str(self.options.ltres)
                    # command += " qtres "             + str(self.options.qtres)
                    # command += " pathomit "          + str(self.options.pathomit)
                    # command += " rightangleenhance " + str(self.options.rightangleenhance).lower()
                    # command += " colorsampling "     + str(self.options.colorsampling)
                    # command += " numberofcolors "    + str(self.options.numberofcolors)
                    # command += " mincolorratio "     + str(self.options.mincolorratio)
                    # command += " numberofcolors "    + str(self.options.numberofcolors)
                    # command += " colorquantcycles "  + str(self.options.colorquantcycles)
                    # command += " layering "          + str(self.options.layering)
                    # command += " strokewidth "       + str(self.options.strokewidth)
                    # command += " linefilter "        + str(self.options.linefilter).lower()
                    # command += " scale "             + str(self.options.scale)
                    # command += " roundcoords "       + str(self.options.roundcoords)
                    # command += " viewbox "           + str(self.options.viewbox).lower()
                    # command += " desc "              + str(self.options.desc).lower()
                    # command += " blurradius "        + str(self.options.blurradius)
                    # command += " blurdelta "         + str(self.options.blurdelta)
                    #
                    # # Create the vector traced SVG file
                    # with os.popen(command, "r") as tracerprocess:
                    #     result = tracerprocess.read()
                    #     if "was saved!" not in result:
                    #         self.msg("Error while processing input: " + result)
                    #         self.msg("Check the image file (maybe convert and save as new file) and try again.")
                    #         self.msg("\nYour parser command:")
                    #         self.msg(command)
                    #
                    #
                    # # proceed if traced SVG file was successfully created
                    # if os.path.exists(exportfile + ".svg"):
                    #     self.fit_svg(exportfile, image)
                    
                    #remove the old image or not                    
                    #TODO: re-enable this?
                    # if self.options.keeporiginal is not True:
                        # image.delete()
                        # for sub_image in detail_sub_images:
                        #     sub_image.delete()
            else:
                self.msg("No images found in selection! Check if you selected a group instead.")      
        else:
            self.msg("Selection is empty. Please select one or more images.")

if __name__ == '__main__':
    try:
        import inkscape_ExtensionDevTools

        inkscape_ExtensionDevTools.inkscape_run_debug()
    except:
        pass
    Continuous_outline().run()