import sys
from pathlib import Path
import matplotlib.pyplot as plt
#from scipy import ndimage as nd
from matplotlib.patches import Rectangle
import numpy as np
from PIL import *


def luminance(img, out):
    """
    Perceived luminance calculation
    """
    # Compare with standard luminance: 0.2126*R + 0.7152*G + 0.0722*B
    lum = 0.299*img[:, :, 0] + 0.589*img[:, :, 1]+0.114*img[:, :, 2]

    plt.figure()
    plt.title("operator box-area")
    plt.imshow(lum)
    imag = Image.fromarray(np.asarray(lum), mode="RGB")
    plt.show()
    if out:
        imag.save(out + "/cut_img.png")
        plt.savefig(out + "/operator_box_area.png")
    return lum


def ESF(lum, out):
    """
    Edge Spread Function calculation
    """
    X = lum[100, :]
    mu = np.sum(X)/X.shape[0]
    tmp = (X[:] - mu)**2
    sigma = np.sqrt(np.sum(tmp)/X.shape[0])
    edge_function = (X[:] - mu)/sigma

    edge_function = edge_function[::3]
    x = range(0, edge_function.shape[0])

    plt.figure()
    plt.title(r'ESF')
    plt.plot(x, edge_function, '-ob')
    plt.show()
    if out:
        plt.savefig(out + "/ESF.png")
    return edge_function


def LSF(esf, out):
    """
    Line Spread Function calculation
    """
    lsf = esf[:-2] - esf[2:]
    x = range(0, lsf.shape[0])

    plt.figure()
    plt.title("LSF")
    plt.xlabel(r'pixel')
    plt.ylabel('intensidad')
    plt.plot(x, lsf, '-or')
    plt.show()
    if out:
        plt.savefig(out + "/LSF.png")
    return lsf


def MTF(lsf, out):
    """
    Modulation Transfer Function calculation
    """
    mtf = abs(np.fft.fft(lsf))
    mtf = mtf[:]/np.max(mtf)
    mtf = mtf[:len(mtf)//2]
    ix = np.arange(mtf.shape[0]) / (mtf.shape[0])
    mtf_poly = np.polyfit(ix, mtf, 6)
    poly = np.poly1d(mtf_poly)

    plt.figure()
    plt.title("MTF")
    plt.xlabel(r'Frequency $[cycles/pixel]$')
    plt.ylabel('mtf')
    p, = plt.plot(ix, mtf, '-or')
    ll, = plt.plot(ix, poly(ix))
    plt.legend([p, ll], ["MTF values", "polynomial fit"])
    plt.grid()
    plt.show()
    if out:
        plt.savefig(out + "/MTF.png")
    return mtf


def mtf_from_img(image, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    lum = luminance(image, out)
    esf = ESF(lum, out)
    lsf = LSF(esf, out)
    mtf = MTF(lsf, out)
    print(mtf)


class ImageAreaSelect(object):
    def __init__(self, img, out, fit=False):
        self.img = img
        self.out = out
        self.fit = fit

        plt.figure()
        plt.title("Testing Image")
        plt.xlabel(r'M')
        plt.ylabel(r'N')
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1, antialiased=True, color='b',
                              linestyle='solid', lw=1.2)
        self.rect.set_fill(False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

        self.draw_rect = False

        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        plt.imshow(self.img, cmap="gray")
        plt.show()

    def on_press(self, event):
        self.draw_rect = True
        self.x0 = int(event.xdata)
        self.y0 = int(event.ydata)

    def on_motion(self, event):
        if self.draw_rect:
            if self.x1 != int(event.xdata) or self.y1 != int(event.ydata):
                self.x1 = int(event.xdata)
                self.y1 = int(event.ydata)
                self.rect.set_width(self.x1 - self.x0)
                self.rect.set_height(self.y1 - self.y0)
                self.rect.set_xy((self.x0, self.y0))
                self.ax.figure.canvas.draw()

    def on_release(self, event):
        self.draw_rect = False
        self.x1 = int(event.xdata)
        self.y1 = int(event.ydata)
        self.calc()

    def calc(self):
        if self.x1 == self.x0 or self.y1 == self.y0:
            return
        print('crop(x,y):', self.x0, self.x1, self.y0, self.y1)
        img = self.img[self.y0:self.y1, self.x0:self.x1]
        mtf_from_img(img, self.out + "/" + str(self.x0) + "_" + str(self.x1) + "_" + str(self.y0) + "_" + str(self.y1))


if __name__ == "__main__":
    print("Usage: mtf.py <image-file> <output-folder>")
    # set defaults
    if len(sys.argv) > 1:
        im_file = sys.argv[1]
    else:
        im_file = "./images/prueba.png"
    if len(sys.argv) > 2:
        out_folder = sys.argv[2]
    else:
        out_folder = "./out"

    im = plt.imread(im_file)
    ImageAreaSelect(im, out_folder, fit=True)
