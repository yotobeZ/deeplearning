def generate_tableau20_colors():
    tableau20_rgb_components = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                                (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                                (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                                (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                                (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    return [(rgb[0] / 255., rgb[1] / 255., rgb[2] / 255.) for rgb in tableau20_rgb_components]
#数组里是20种颜色，代表什么？
#（31/255. ， 119/255. ， 180/255.）像素值除以255是个啥玩意
#两个解释：
# 1.RGB的颜色格式就是这，“/”不理解为除
# 2.“/”理解为除，也是色彩的值，不除之前计算机用字节char表示变量并表示的0~255;除之后用浮点数float或double变量表示