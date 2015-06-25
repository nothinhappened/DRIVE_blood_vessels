function main
    %distcomp.feature('LocalUseMpiexec',false);
    %matlabpool('open',4);
    global MAXD; %Max thickness for vessels - lower filters more (Strongest impact on performance, time to process scales with square of MAXD)
    global TSTART; %Minimum threshold
    global TEND; %Maximum threshold
    global MAXPHI; %Maximim angle (phi) for vessel - higher filters more(see paper)
    global PSIZE; %Minimum size for connected component
    global CMIN; %Minimum contrast for component
    global CALIBRATE; %enable output of binarized images;
    global PARTIALS; %enable output of partials. Recommended.
    MAXD = 9; %Default 9
    TSTART=0.3; %Default 0.02
    TEND=0.8; %Default 0.7
    PSIZE=44; %Default 44
    MAXPHI = 140; %Default 135
    CMIN=1.07; %Default 1.05   

    iterations=45; %How many steps between minimum and maximum threshold, higher = better results, performance scales linearly with iterations

    
    global NOVERIFY %Turn verification on and off.
    NOVERIFY=0;
    CALIBRATE=0;
    PARTIALS=1;
    MASK = 0; %Use mask to speed computation
    global fname; %file name global variable
    
    
    %im=imread('retina_small.jpeg','jpeg');
    %im=im(:,:,2);
    %im=demask(im);
    
    im=imread('DRIVE\DRIVE\test\images\02_test.tif','tif');
    green = im(:,:,2);
    if(MASK==1)
        mask = imread('DRIVE\DRIVE\test\mask\02_test_mask.gif','gif');
        mask = imcomplement(mask);
        im= mask + green;
    else
        im=green;
    end
    
    imwrite(uint8(im),'Image.tiff','tiff','Compression','none');
    %im = uint8(equalize(im));
    imwrite(uint8(im),'ImageEq.tiff','tiff','Compression','none');
    
    output = runEach(iterations,im);
    imwrite(uint8(output)*255,strcat('27-final-45iterations.tiff'),'tiff','Compression','none')
    
    
    %matlabpool('close');
    %output = im2bw(im,.2);
    %
    %
    %
    %output = verify(output);
       
        %temp = im2bw(im, .3);
        %subplot(211);
        %imshow(temp);
        %subplot(212);
        %temp = verify(temp);
        %imwrite(uint8(temp)*255,strcat(num2str(i),'out.tiff'),'tiff','Compression','none');
        
    imshow(output);
    
end

% function out=maskImage(image) 
% This was used to mask images which did not have a binary mask - ie. images I downloaded and resized.
%     [X,Y,Z] = size(image);
%     for x=1:X
%         for y=1:Y
%             if (image(x,y)<25)
%                 image(x,y)=255;
%             end
%         end
%     end
%     out=image;
% end

function output= runEach(iterations,im)
    global TEND;
    global TSTART;
    global NOVERIFY;
    global CALIBRATE;
    global PARTIALS;
    [X,Y,Z] = size(im);
    output = zeros(X,Y);
    jmp = double((TEND-TSTART))/double(iterations);
    for i=0:iterations
        cur = TSTART + double(jmp) * double(i);
        temp = im2bw(im, cur);
        %imwrite(uint8(im),strcat(num2str(i),'image.tiff'),'tiff','Compression','none');
        %imwrite(uint8(temp)*255,strcat(num2str(i),'binarized.tiff'),'tiff','Compression','none');
        if(CALIBRATE==1) 
            imwrite(uint8(temp)*255,strcat(num2str(cur*100),'binarized.tiff'),'tiff','Compression','none') 
        end
        if(NOVERIFY==0) temp = verify(temp,im); end
       
        %temp = verifyContrast(temp, im);
        %imwrite(uint8(temp)*255,strcat(num2str(i),'temp.tiff'),'tiff','Compression','none');
        
        temp = imcomplement(temp);
        output = output | temp;
        if(PARTIALS==1)
            imwrite(uint8(output)*255,strcat(num2str(i),'partial.tiff'),'tiff','Compression','none');
        end
    end
    output = imcomplement(output);
end

% Function for verifying contrast based on region, this is unused.
% function im = verifyContrast(bin,im)
%     global CMIN;
%     [X,Y,Z] = size(bin);
%     for x=1:X
%         for y=1:Y
%             if(bin(x,y)==0)
%                 %We have a vessel pixel
%                 Ip=im(x,y); %intensity of our pixel
%                 Inmax=0;
%                 for xsub=-1:1
%                     for ysub=-1:1
%                         %For each neighbour pixel in 8 neighbourhood
%                         In=im(x+xsub,y+ysub);
%                         if(In > Inmax)
%                             Inmax = In %Get max value for neighbouring pixel
%                         end
%                     end
%                 end
%                 c = Ip/Inmax;
%                 if(c<CMIN) %If we don't have enough contrast
%                     im(x,y)=1;
%                 end
%             end
%         end
%     end
% end


function closeMap=getCloseMap(image)
    [X,Y,Z] = size(image);
    closeMap = zeros(X,Y,2);
    
    for x=1:X
        %dummy = zeros(Y,2);
        for y=1:Y
            %dummy(y,1:2)=getNearestPixel(image,x,y);
            [xloc,yloc]=getNearestPixel(image,x,y);
            closeMap(x,y,1)= xloc;
            closeMap(x,y,2)= yloc;
        end
        %closeMap(x,:,:)=dummy;
    end
end


function out = verify(image,original)
    global MAXD;
    global MAXPHI;
    global PSIZE;
    global CMIN;
    disp('Verify started');
    out = image;
    if(containsWhite(image)==0) %Our image is all black, return)
        return;
    end
    [X,Y] = size(image);
    
    closeMap = getCloseMap(image);
    disp('map generated');
    for x=1:X
        for y=1:Y
            %for each pixel
            if(image(x,y)==1) %If its white, we don't care 
                continue;
            end
            %If pixel is black, we need to prune
            %First, get information about the vector to our nearest pixel
            
            xloc=closeMap(x,y,1);
            yloc=closeMap(x,y,2);
            if(xloc == -1 && yloc==-1)
                out(x,y)=1;
                continue;
            end
            
            d=0;
            phi=0;
            Imax=uint8(1);
            %Now we de our calculation for each neighbouring pixel
            for xsub = -1:1
                for ysub = -1:1
                    if(x+xsub < 1 || y+ysub<1)
                        continue;
                    end
                    
                    %vector information for nearest pixel
                    xloc2=closeMap(x+xsub,y+ysub,1);
                    yloc2=closeMap(x+xsub,y+ysub,2);

                    if(xloc2==-1 || yloc2==-1)
                        continue;
                    end
                    
                    dtemp = getDistance(xloc,yloc,xloc2,yloc2);
                    phitemp = getAngle(x,y,xloc,yloc,xloc2,yloc2);
                    I = original(xloc2,yloc2);
                    
                    %We keep the information if it is a maxima
                    d=max([d,dtemp]);
                    phi=max([phi,phitemp]);
                    Imax=max([I,Imax]);
                    
%                     if(dtemp > d)
%                         d=dtemp;
%                     end
%                     if(phitemp > phi)
%                         phi=phitemp;
%                     end
%                     if(I>Imax)
%                         I=Imax;
%                     end
                end
            end
            
            %Now we prune based on d and phi
            %These are pixel based pruning operations
            %disp(Imax/double(original(x,y)));
            %disp(double(Imax)/(double(original(x,y)+1)));
            if(d>MAXD || abs(phi) < MAXPHI || double(Imax)/(double(original(x,y)+1)) < CMIN)
                out(x,y)=1;
            end
        end
    end
    
    %Then we need to prune entire image based on connected component size
    [L,num] =bwlabel(imcomplement(out),4);
    count = zeros(num+1,1);
    for x =1:X
        for y=1:Y
            count(L(x,y)+1) = count(L(x,y)+1) + 1;
        end
    end
    
    %prune small sized parts
    for x =1:X
        for y=1:Y
            if (count(L(x,y)+1)<PSIZE)
                out(x,y)=1;
            end
        end
    end
end

% function in=inBounds(X,Y,x,y)
% %This is left here in case I'd like to start using it again. Profiling showed ot took up a huge amount of processing. Runtime exception handling is being used instead.
%     in=1;
%     if( x<1 || y<1 || x>X || y>Y )
%         in=0;
%     end
% end

function dist=getDistance(x1,y1,x2,y2)
    x=x1-x2;
    y=y1-y2;
    
    dist = sqrt(x*x+y*y);
end

function out=containsWhite(image)
    [X,Y] = size(image);
    out=0;
    for x = 1:X
        for y = 1:Y
            if(image(x,y)==1)
                out=1;
                return;
            end
        end
    end
end


function [xloc,yloc] = getNearestPixel(image,x,y)
    %A return of -1 means always prune
    global MAXD;
    dout = 1;
    dfinal=32767;%intmax('int32'); Calls to intmax took too long, I've decided to support no more than 16 bit image width/height
    dmin =32767; %intmax('int32');
    
    try
        if(image(x,y)==1)
            xloc=x;
            yloc=y;
            return;
        end
    catch err
    end
    
    while(dout <= dfinal)
        for d=-1*dout:dout
            for l=0:3
                if(l==0)
                    dx=dout;
                    dy=d;
                elseif(l==1)
                    dx=-1*dout;
                    dy=d;
                elseif(l==2)
                    dx=d;
                    dy=-1*dout;
                elseif(l==3)
                    dx=d;
                    dy=dout;
                else
                    disp('Fatal error in getNearestPixel()');
                end
            
                %For each pixel in our square region
                
                %This operation is inlined for performance, this would be
                %functionalized in a 'normal' programming language but
                %since matlab is pass-by-val and won't inline, this saves
                %us several million memory allocations.
                try
                    if(image(x+dx,y+dy)==1)
                        %if we have a white pixel
                        if(dfinal==32767)
                            %if we haven't found a white pixel yet
                            dfinal=dout*sqrt(2); %we want to stop once we've reached twice our dout;
                            dmin = getDistance(x,y,x+dx,y+dy);
                            xloc = x+dx;
                            yloc = y+dy;
                        else
                            %we have found our distance already, is this one
                            %better?
                            dtemp = getDistance(x,y,x+dx,y+dy);
                            if(dtemp < dmin)
                                %it is a better distance
                                dmin=dtemp;
                                xloc = x+dx;
                                yloc = y+dy;
                            end
                        end
                    end
                catch err
                    continue;
                end
            end
        end
        if(dout>=MAXD*sqrt(2) && dfinal==32767)
            xloc = -1; 
            yloc = -1;
            return;
        end
        dout = dout + 1;
    end
end

function phi=getAngle(x1,y1,x2,y2,x3,y3)
    pepDist = getDistance(x1,y1,x2,y2);
    penDist = getDistance(x1,y1,x3,y3);
    top = dot([x1-x2,y1-y2],[x1-x3,y1-y3]);
    phi = (180/pi) * acos(top/(penDist*pepDist));
end

function out=equalize(im)
    [N,M] = size(im);
    colorspace = 256;
    histg = zeros(1,colorspace);
    chist = zeros(1,colorspace);
    table = zeros(1,colorspace);
    image = zeros(N,M);
    

    for i=1:N
        for j=1:M
            %This is normal control flow
            histg(im(i,j)+1) =histg(im(i,j)+1)+1 ;
        end

    end
    %We our out of the while loop and have read the image into our histogram

    %Create the cummulative histogram
    chist(1) = histg(1);
    for i=2:256
        chist(i) = histg(i) + chist(i-1);
    end

    %Create our lookup table for values
    for i=1:256
        table(i) = round((255/(N*M))*chist(i));
    end


    %Create the new image
    for i=1:N
        for j=1:M
            image(i,j) = table(im(i,j)+1);
        end
    end
    out = image;
end