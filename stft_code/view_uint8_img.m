function view_uint8_img(img)
    figure;
    imshow(uint8((img-min(min(img)))/(max(max(img))-min(min(img)))*255));
end