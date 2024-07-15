from cornea.segment.interface import get_interface_line

def create_hole(image, mask):


    interface_line = get_interface_line(mask)


    hole = image.clone()

    
    return hole

