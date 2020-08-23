import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import *


def luminance(img, out, show):
    """
    Perceived luminance calculation
    """
    # Compare with standard luminance: 0.2126*R + 0.7152*G + 0.0722*B
    lum = 0.299*img[:, :, 0] + 0.589*img[:, :, 1]+0.114*img[:, :, 2]

    plt.figure()
    plt.title("operator box-area")
    plt.imshow(lum)
    imag = Image.fromarray(np.asarray(lum), mode="RGB")
    if out:
        imag.save(out + "/cut_img.png")
        plt.savefig(out + "/operator_box_area.png")
    if show:
        plt.show()
    else:
        plt.close()
    return lum


def ESF(lum, out, show):
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
    if out:
        plt.savefig(out + "/ESF.png")
    if show:
        plt.show()
    else:
        plt.close()
    return edge_function


def LSF(esf, out, show):
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
    if out:
        plt.savefig(out + "/LSF.png")
    if show:
        plt.show()
    else:
        plt.close()
    return lsf


def MTF(lsf, out, show):
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
    if out:
        plt.savefig(out + "/MTF.png")
    if show:
        plt.show()
    else:
        plt.close()
    return mtf


def mtf_from_img(image, out, show):
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
    lum = luminance(image, out, show)
    esf = ESF(lum, out, show)
    lsf = LSF(esf, out, show)
    mtf = MTF(lsf, out, show)
    return mtf


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
        mtf_from_img(img, self.out + "/" + str(self.x0) + "_" + str(self.x1) + "_" + str(self.y0) + "_" + str(self.y1), True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mtf.py image-file [optional-crop list:]x0,x1,y0,y1")

    # set args values or reset to defaults
    if len(sys.argv) > 1:
        im_file = sys.argv[1]
    else:
        im_file = "./images/prueba.png"
    im = plt.imread(im_file)
    out_folder = "./out"

    if len(sys.argv) > 2:
        for i in range(len(sys.argv) - 2):
            crop = list(map(int, sys.argv[2+i].split(',')))
            x0 = int(crop[0])
            x1 = int(crop[1])
            y0 = int(crop[2])
            y1 = int(crop[3])
            img = im[y0:y1, x0:x1]
            res = mtf_from_img(img, out_folder + "/" + str(x0) + "_" + str(x1) + "_" + str(y0) + "_" + str(y1), False)
            print(str(list(res))[1:-1])
    else:
        ImageAreaSelect(im, out_folder, fit=True)
