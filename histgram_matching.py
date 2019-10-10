# -*- coding: utf-8 -*-
# histgram maching (only for grayscale image)
#
# usage:
# $python histgram_matching.py source_img_name reference_img_name output_img_name
#
# algorithm:
# https://en.wikipedia.org/wiki/Histogram_matching
#
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


#compute histgram (pdf) and cdf
# pdf: probability density function     : 確率密度関数
# cdf: Cumulative distribution function : 累積分布関数
#
#  input is supporse to be a grayscale image
def calc_pdf_cdf ( img ) :
	height = img.shape[0]
	width  = img.shape[1]

	pdf = cv2.calcHist([img],[0],None,[256],[0,256] )
	pdf /= (width*height)
	pdf = np.array( pdf )

	cdf = np.zeros( 256 )
	cdf[0] = pdf[0]
	for i in range(1,256) :
		cdf[i] = cdf[i-1] + pdf[i]

	return pdf, cdf



srcimg_name = sys.argv[1]
refimg_name = sys.argv[2]
outimg_name = sys.argv[3]

#画像を読み込み、グレースケール化し、float型に変換
src_img = cv2.cvtColor( cv2.imread( srcimg_name ), cv2.COLOR_RGB2GRAY)
ref_img = cv2.cvtColor( cv2.imread( refimg_name ), cv2.COLOR_RGB2GRAY)
cv2.imwrite("img11.png",src_img )

#calc pdf and cdf
src_pdf, src_cdf = calc_pdf_cdf(src_img)
ref_pdf, ref_cdf = calc_pdf_cdf(ref_img)

#calp mapping new_gray_level = mapping(old_gray_level)
mapping = np.zeros(256, dtype = int)

for i in range(256) :
	#search j such that src_cdf(i) = ref_cdf(j)
	# and set mapping[i] = j
	for j in range(256) :
		if ref_cdf[j] >= src_cdf[i] :
			break
	mapping[i] = j

#gen output image
out_img = np.zeros_like(src_img, dtype = np.uint8)
for i in range(256) :
	out_img[ src_img == i ] = mapping[i]
cv2.imwrite( outimg_name, out_img);


# check histgrams (これ以降はただのデバッグ)
out_pdf, out_cdf = calc_pdf_cdf(out_img)
print(src_pdf)
plt.plot(src_cdf, color = 'r')
plt.plot(ref_cdf, color = 'g')
plt.plot(out_cdf, color = 'b')
plt.xlim([0,256])
plt.show()
