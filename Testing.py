import helpers.edge_detection as edge

test = list(edge.detect_image_contours('./sub_image_0.png'))
test2 = list(edge.detect_image_contours('./sub_image_1.png'))

test.extend(test2)
print(test[0])