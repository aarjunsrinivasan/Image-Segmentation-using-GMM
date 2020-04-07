clc
Image = imread('C:\Users\Praveen\ENPM 673 Project 3\frame199.jpg');
[width, height, channel] = size(Image) ;
[X,Y] = meshgrid(1:height,1:width) ;
imshow(Image) ;
hold on
[cx,cy] = getpts ;   % click at the center and approximate Radius
r = sqrt(diff(cx).^2+diff(cy).^2) ;
theta = linspace(0,2*pi) ;
xc = cx(1)+r*cos(theta) ; 
yc = cy(1)+r*sin(theta) ; 
plot(xc,yc,'r') ;
% Keep only points lying inside circle
idx = inpolygon(X(:),Y(:),xc',yc) ;
for i = 1:channel
    I1 = Image(:,:,i) ;
    I1(~idx) = 255 ;
    Image(:,:,i) = I1 ;
end
figure
imshow(Image)