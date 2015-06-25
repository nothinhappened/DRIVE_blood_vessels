function drive_optimized_completely_unreadable
    distcomp.feature('LocalUseMpiexec',false);
    matlabpool('open',4);
    %     global MAXD; %Max thickness for vessels - lower filters more (Strongest impact on performance, time to process scales with square of MAXD)
    %     global TSTART; %Minimum threshold
    %     global TEND; %Maximum threshold
    %     global MAXPHI; %Maximim angle (phi) for vessel - higher filters more(see paper)
    %     global PSIZE; %Minimum size for connected component
    %     global CMIN; %Minimum contrast for component
    %     global IOCALIBRATE; %calibration from gui
    %     global CALIBRATE; %enable output of binarized images;
    %     global PARTIALS; %enable output of partials. Recommended.
    %     global USEMASK;
    %     global MASKTYPE;
    %     global ITERATIONS;


    %global threshImage; %Used to try to get some optimization in the app,
    %not used within this function, but used elsewhere.
    %     global original;
    %     global closeMap;
    %     global mask;



    MAXD = 9; %Default 9
    TSTART=0.2; %Default 0.2
    TEND=0.7; %Default 0.7
    PSIZE=44; %Default 44
    MAXPHI = 135; %Default 135
    CMIN=1.07; %Default 1.09
    IOCALIBRATE=1;
    EQUALIZE=1;

    ITERATIONS=45; %How many steps between minimum and maximum threshold, higher = better results, performance scales linearly with iterations

    NOVERIFY=0;%Turn verification on and off.
    CALIBRATE=1;
    PARTIALS=1;


    %After much testing, the mask makes only a minimal difference on the
    %result, and a much larger difference on performance. Best options are
    %MASKTYPE=0, USEMASK=1
    MASKTYPE=0; %Ignored if USEMASK is 0. 0 for image based, 1 for map based. 1 is untested, but should result in a slightly better image
    USEMASK = 1; %Use mask to speed computation



    %im=imread('retina_small.jpeg','jpeg');
    %im=im(:,:,2);
    %im=demask(im);

    original=imread('DRIVE\DRIVE\test\images\09_test.tif','tif');
    imwrite(uint8(original),'image09/Image.tiff','tiff','Compression','none');
    original = original(:,:,2);
    
    if(USEMASK==1 && MASKTYPE==0&&~EQUALIZE)
        mask = imread('DRIVE\DRIVE\test\mask\09_test_mask.gif','gif');
        mask = imcomplement(mask);
        original= mask + original;
    elseif(USEMASK==1 && MASKTYPE==0)
        mask = imread('DRIVE\DRIVE\test\mask\09_test_mask.gif','gif');
        mask = imcomplement(mask);
        [X,Y]=size(original);
        maxVal=0;
        for x=1:X
            for y=1:Y
                if(original(x,y)>maxVal)
                    maxVal=original(x,y)+2;
                end
            end
        end
        %there isn't any easy matrix op to do  this, since we are doing a
        %replace. Here we go doing it elementwise
        for x=1:X
            for y=1:Y
                if(mask(x,y)>0)
                    original(x,y)=maxVal;
                end
            end
        end 
        
        %disp((mask>0).*maxVal);
        %disp(maxVal);
        %mask = uint8(mask>0)*maxVal;
        %original=  mask + original;
    elseif (USEMASK==1)
        mask = imread('DRIVE\DRIVE\test\mask\09_test_mask.gif','gif');
    end
    
    
    imwrite(uint8(original),'image09/green.tiff','tiff','Compression','none');
    if(EQUALIZE)
        original = uint8(equalize(original));
    end
    imwrite(uint8(original),'image09/ImageEq.tiff','tiff','Compression','none');

    if(IOCALIBRATE==1)
        [TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS]=calibrate(TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS,original);
        disp(sprintf('Using\tTSTART %3f\t TEND %d\n\t\tIterations %3f',TSTART,TEND,ITERATIONS));
    end;
    


    [X,Y] = size(original);
    closeMap = zeros(X,Y,2);



    %output = runEach(ITERATIONS);
    %function output= runEach(iterations)

    [X,Y,Z] = size(original);
    output = zeros(X,Y);
    jmp = double((TEND-TSTART))/double(ITERATIONS);
    for i=0:ITERATIONS
        cur = TSTART + double(jmp) * double(i);
        threshImage = im2bw(original, cur);
        disp(sprintf('Starting binarization %f\n', cur));
        %imwrite(uint8(im),strcat(num2str(i),'image.tiff'),'tiff','Compression','none');
        %imwrite(uint8(temp)*255,strcat(num2str(i),'binarized.tiff'),'tiff','Compression','none');
        if(CALIBRATE==1)
            disp('Calibration image created');
            imwrite(uint8(imcomplement(threshImage))*255,strcat('image09/',strcat(num2str(cur*100),'binarized.tiff')),'tiff','Compression','none')
        end
        if(NOVERIFY==0)
            %okay, at this point, no one is going to read these comments
            %because this function is too big. Below you will read that the
            %worst thing I've ever done is inline that last function, but
            %this is worse. When I inline runeach that will become the
            %worst, but for now - seriously what am I doing...
            %function verify()

            if(containsWhite(threshImage)==0) %Our image is all black, return)
                return;
            end
            [X,Y] = size(threshImage);
            

            

            %closeMap = getCloseMap();

            disp('Creating distance map');
            %Inlined for optimization
            %function closeMap=getCloseMap()
            parfor x=1:X
                for y=1:Y
                    xloc=-1;
                    yloc=-1;
                    if((USEMASK ==1) && (MASKTYPE==1))
                        if(mask(x,y)==0)
                            closeMap(x,y,:) = [x,y];
                            continue;
                        end
                    end
                    %[xloc,yloc]=getNearestPixel(x,y);

                    %Ugliest thing I have EVER done while coding.... this ugly
                    %beast is being inlined
                    %function [xloc,yloc] = getNearestPixel(x,y)
                    %A return of -1 means always prune

                    dout = 1;
                    dfinal=32767;%intmax('int32'); Calls to intmax took too long, I've decided to support no more than 16 bit image width/height
                    dmin =32767; %intmax('int32');
                    cont=1;
                    try
                        if(threshImage(x,y)==1)
                            xloc=x;
                            yloc=y;
                            cont=0;
                        end
                    catch err
                    end

                    while(dout <= dfinal && cont)
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
                                    dx=d;
                                    dy-d;
                                end

                                %For each pixel in our square region

                                %This operation is inlined for performance, this would be
                                %functionalized in a 'normal' programming language but
                                %since matlab is pass-by-val and won't inline, this saves
                                %us several million memory allocations.
                                try
                                    if(threshImage(x+dx,y+dy)==1)
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
                        if(dout>=MAXD && dfinal==32767)
                            xloc = -1;
                            yloc = -1;
                            cont=0;
                        elseif(dout>=MAXD)
                            cont=0;
                        end
                        dout = dout + 1;
                    end


                    closeMap(x,y,:)= [xloc,yloc];
                end
                %closeMap(x,:,:)=dummy;
            end
            %end function closeMap

            disp('map generated');
            disp('Verify started');
            parfor x=1:X
                for y=1:Y
                    %for each pixel
                    if(threshImage(x,y)==1) %If its white, we don't care
                        continue;
                    end
                    %If pixel is black, we need to prune
                    %First, get information about the vector to our nearest pixel

                    xloc=closeMap(x,y,1);
                    yloc=closeMap(x,y,2);
                    if(xloc == -1 && yloc==-1)
                        threshImage(x,y)=1;
                        continue;
                    end

                    d=0;
                    phi=0;
                    Imax=uint8(1);
                    %Now we de our calculation for each neighbouring pixel
                    for xsub = -1:1
                        for ysub = -1:1
                            if(x+xsub < 1 || y+ysub<1 || x+xsub > X || y+ysub>Y)
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
                        threshImage(x,y)=1;
                    end
                end
            end

            %Then we need to prune entire image based on connected component size
            [L,num] =bwlabel(imcomplement(threshImage),4);
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
                        threshImage(x,y)=1;
                    end
                end
            end
            disp('verify complete');
        end


        %temp = verifyContrast(temp, im);
        %imwrite(uint8(temp)*255,strcat(num2str(i),'temp.tiff'),'tiff','Compression','none');

        threshImage = imcomplement(threshImage);
        output = output | threshImage;
        if(PARTIALS==1)
            imwrite(uint8(output)*255,strcat('image09/',strcat(num2str(i),'partial.tiff')),'tiff','Compression','none');
        end
    end
    output = imcomplement(output);



    imwrite(uint8(output)*255,strcat('image09/03-final-45iterations.tiff'),'tiff','Compression','none')


    matlabpool('close');
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




function [TSTART,TEND,ITERATIONS] = calibrateThreshold(original,TSTART,TEND)
    
    temp = im2bw(original,TSTART);
    imshow(temp);
    iteration = .1;
    yn=input('A minimum and maximum threshold must be selected for processing.\nFor the minumum, the ideal image will have a couple of vessels be barely visible.\nIf the image is just a white circle, then it should not be used (select Too few vessels)\nPlease choose whether this image matches that description\nOptions:\n(l) Less Vessels (Image has big black blobs)\n(m) More vessels(Image is just a white circle)\n(j) Just right (Image has some vessels or pixels, but no blobs)\n','s');
    lastchosen = yn;
    while(strcmpi(yn,'j')==0)
        if(strcmpi(yn,'l'))
            if(TSTART==0)
                %If the person needs it lighter than 0, just exit.
                yn='j';
                continue;
            end
            if(~strcmpi(yn,lastchosen))
                lastchosen = yn;
                iteration = iteration/2;
            end
            TSTART = TSTART - iteration;
            if(TSTART<0)
                TSTART=0;
            end
            
        end
        if(strcmpi(yn,'m'))
            if(~strcmpi(yn,lastchosen))
                lastchosen = yn;
                iteration = iteration/2;
            end
            TSTART = TSTART + iteration;
        end
        
        temp = im2bw(original,TSTART);
        imshow(temp);
        yn=input('How about this image?\nOptions:\n(l) Less Vessels (Image has big black blobs)\n(m) More vessels(Image is just a white circle)\n(j) Just right (Image has some vessels or pixels, but no blobs)\n','s');
    end
    
    tendDefault = TEND;
    temp = im2bw(original,TEND);
    imshow(temp);
    iteration = .1;
    disp('The maximum threshold for processing must contain all vessels which need to be segmented.\n If the threshold is too high, processing will take a long time, regardless of the number of iterations selected.\nPlease choose this carefully, so that it includes all parts needed, but is not a fully black circle.\nThe final image will likely contain a white spot around only the optic nerve. For performance, leaving this area blank (white) is best.\nIf the vessels for this are are needed, then include them');
    yn=input('Options\n(m) There are not enough vessels in this image - too much white\n(l)This image has too much black/I need better performance\n(j)This image is perfect\n','s');
    lastchosen=yn;
    while(strcmpi(yn,'j')==0)
        if(strcmpi(yn,'l'))
            if(~strcmpi(yn,lastchosen))
                lastchosen = yn;
                iteration = iteration/2;
            end
            TEND = TEND - iteration;
            
        elseif(strcmpi(yn,'m'))
            if(~strcmpi(yn,lastchosen))
                lastchosen = yn;
                iteration = iteration/2;
            end
            TEND = TEND + iteration;
        elseif(strcmpi(yn,'s'))
            iteration = .1;
            TEND =tendDefault;
        end
        temp = im2bw(original,TEND);
        imshow(temp);
        yn=input('Options\n(m) There are not enough vessels in this image - too much white\n(l)This image has too much black/I need better performance\n(j)This image is perfect\n(s)Start Over\n','s');
    end
    m = uint8((TEND-TSTART)*255);
    disp(sprintf('The number of iterations should also be optimized.\nFor best results, 1 image per iteration can be selected (%d iterations).\nIterations can take as long as 30 minutes each for the later, darker iterations, and as short as 10 seconds early on\nMore iterations helps with vessel detection.\nHow many iterations would you like(Minimum 5, Max %d)?\nDont know what to put? Try using a fraction of the max value, such as %d or %d, or just use %d and wait',m,m,m/2,m/3));
    ITERATIONS = input('Options - Any positive integer between min and max value\n');
    
end
    
function [TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS]=calibrateAll(original,TSTART,TEND)  
    yn = input('Use assistance for image thresholding?\nOptions:\n(y) Yes\n(n) No\n','s');
    if(strcmpi('y',yn)||strcmpi('yes',yn))
       [TSTART,TEND,ITERATIONS]=calibrateThreshold(original,TSTART,TEND); 
    else
        TSTART = input('Enter value for starting threshold (default: 0.3)\nOptions - real number value [0-.99]\n');
        %Again matlab, I get that %lf is not c-standard until c98 or
        %somewhere around there, but please work with me when I use %lf,
        %its pretty commonly used now. Figure it out, don't error out.
        TEND = input(sprintf('Enter value for end threshold (default: 0.7)\nOptions - real number value [%3f-1.0]\n',TSTART));
        ITERATIONS = input(sprintf('Enter value for iterations\nMinimum 5, Maximum %d\n', uint8((TEND-TSTART)*255)));
    end
    
    MAXD = input('Enter MAXD (Default 9)\nMax thickness for vessels - lower filters more (Strongest impact on performance, time to process scales with square of MAXD)\n');
    MAXPHI = input('Enter MAXPHI (Default 140)\nMaximim angle (phi) for vessel - higher filters more(see paper)\nOptions - integer value [0-180]\n');
    PSIZE = input('Enter PSIZE (Default 44)\nMinimum size for connected components\n');
    CMIN = input('Enter minimum contrast (Default 1.09)\nValue is the minimum contrast between a pixel and the background which allows it to be considered a vessel\nValues below 1.25 recommended.\nOptions - real number value [1.0-128.0]\n');
    CALIBRATE = input('Allow saving of thresholded images used for reference later?\nOptions:\n(1) Yes\n(0)No\n');
    PARTIALS = input('Allow saving of partial results (recommended)\nOptions\n(1) Yes\n(0)No\n');
    
    disp('For more options, configure manually in the application.\nAll major calibrations completed');
end

function [TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS]= calibrate(TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS,original)
    in = input('Please select one of the following options by entering the key entered in the brackets\n(d) Use defaults (all automatic)\n(s) Assisted calibration\n(a) Advanced(All manual)\n','s');
    %Why in the world matlab would break from the C standard of '0 means
    %equal, 1 is gt, -1 is lt' is beyond me, but it took a while of
    %debugging before I realized it...
    if(strcmpi(in,'d')) 
        return;
    elseif(strcmpi(in,'a')||strcmpi(in,'all'))
        [TSTART,TEND,MAXD,MAXPHI,PSIZE,CMIN,CALIBRATE,PARTIALS,ITERATIONS]=calibrateAll(original,TSTART,TEND);
    elseif(strcmpi(in,'s'))
        [TSTART,TEND,ITERATIONS]=calibrateThreshold(original,TSTART,TEND);
        in = input('Typically, partial results are saved to the folder the application is run from.\nThese results are there so that the user can exit early if results are taking to long, but still retain some of the information from the session.\nUse partial results?\nOptions:\n(y) Yes\n(n) No\n','s');
        if(strcmpi(in,'y')||strcmpi(in,'yes'))
           PARTIALS=1;
        else
           PARTIALS=0;
        end
        in = input('Would you like a copy of the binarized images used saved to the folder?\nThese can be used later to explain why major vessels do not appear as they should.\nOptions:\n(y) Yes\n(n) No\n','s');
        if(strcmpi(in,'y')||strcmpi(in,'yes'))
           CALIBRATE=1; %Calibration is a historic name for it, it is what I used to calibrate back before the GUI calibration
        else
           CALIBRATE=0;
        end
    end
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

function out=containsWhite(threshImage)
    [X,Y] = size(threshImage);
    out=0;
    for x = 1:X
        for y = 1:Y
            if(threshImage(x,y)==1)
                out=1;
                return;
            end
        end
    end
end




function phi=getAngle(x1,y1,x2,y2,x3,y3)
    pepDist = getDistance(x1,y1,x2,y2);
    penDist = getDistance(x1,y1,x3,y3);
    top = dot([x1-x2,y1-y2],[x1-x3,y1-y3]);
    phi = (180/pi) * acos(top/(penDist*pepDist));
end



function histg=getHist(im)
    histg = zeros(1,256);
    [N,M]=size(im);
    for i=1:N
        for j=1:M
            %This is normal control flow
            histg(im(i,j)+1) =histg(im(i,j)+1)+1 ;
        end
    end
end

%I experimented with histogram equalization at one point, it was horrible.
function out=equalize(im)
    [N,M] = size(im);
    colorspace = 256;
    
    chist = zeros(1,colorspace);
    table = zeros(1,colorspace);
    image = zeros(N,M);
    
    histg=getHist(im);

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
    disp(histg);
    disp(getHist(image));
end