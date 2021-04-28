%% Testing the functions

load Example_Data


RSA_Data = RSA_Data_Enc;

ROI = {'OCC','TMP','FRT','PRT'};
Dim_names = {};

[a,b] = unique(TrialInfo(1:64,6),'stable');
Dim_names{1,1} = TrialInfo([b;b+64],6);
Dim_names{1,2} = [8.5; 8.5];
[a,b] = unique(TrialInfo(1:64,8),'stable');
Dim_names{2,1} = TrialInfo([b;b+64],8);
Dim_names{2,2} = [4.5:4:12.5; 4.5:4:12.5];

col1 = [0.8 0.0 0.0];
col2 = [0.0 0.0 0.8];

imExMDS = Images(1:8:end,3);
imExMDS([1:4 9:12],2) = {uint8(cat(3, 256*col1(1)*ones(70,70), 256*col1(2)*ones(70,70), 256*col1(3)*ones(70,70)))};
imExMDS([5:8 13:16],2) = {uint8(cat(3, 256*col2(1)*ones(70,70), 256*col2(2)*ones(70,70), 256*col2(3)*ones(70,70)))};
plotpictures = true;

add_centroid = true;

r = 2;

TimeVec = RSA_Data.TimeVec;
CurRSA = RSA_Data.(ROI{r}).red16_Data;

RSA_Mat = squeeze(mean(CurRSA,1));
tmp = sort(RSA_Mat(:));
rsa_c_limits = [tmp(find(tmp > 0,1,'first'))*0.95 tmp(end)*0.9];

dims = 2;
MDS_Mat = zeros(size(RSA_Mat,2),dims,length(TimeVec));
for tp = 1:length(TimeVec)
    MDS_Mat(:,:,tp) = cmdscale(squeeze(RSA_Mat(tp,:,:)) + squeeze(RSA_Mat(tp,:,:))',dims);
end
mds_x_limits = [min(min(squeeze(MDS_Mat(:,1,:)))) max(max(squeeze(MDS_Mat(:,1,:))))];
mds_y_limits = [min(min(squeeze(MDS_Mat(:,2,:)))) max(max(squeeze(MDS_Mat(:,2,:))))];

chi2_95perc = chi2inv(0.80,2);
t = linspace(0, 2*pi, 50);
mds_dim1 = 1;
mds_dim2 = 2;

Hyp_Mat = cat(3, Semantic_Mat_red16 > 0, Semantic_Mat_red16 < 0);

ts_data = zeros(size(CurRSA,1), size(CurRSA,2), size(Hyp_Mat,3));
for ts = 1:size(Hyp_Mat,3)
    tmp_hyp = Hyp_Mat(:,:,ts);
    ts_data(:,:,ts) = mean(CurRSA(:,:,tmp_hyp(:)),3);
end

X = kron([1;2;1;2], ones(4,1));

posfig = [200 70 1300 900];
h = figure('Pos', posfig);

% Plot RDM
pos1 = [100 340 450 450];
ax1 = axes('Units','pixels','Pos',pos1);
cur_data = squeeze(RSA_Mat(1,:,:));
pl_im = imagesc(squeeze(RSA_Mat(1,:,:)));
clb = colorbar('Units','pixels','Position',[pos1(1)-30 pos1(2) 20 pos1(4)]);
grid on
set(get(gca,'Yruler'),'Minortick',Dim_names{2,2}(1,:))
set(get(gca,'Xruler'),'Minortick',Dim_names{2,2}(1,:))
set(gca,'clim',rsa_c_limits,'xtick',Dim_names{1,2}(1,:),'xticklabels',[],...
        'ytick',Dim_names{1,2}(2,:),'yticklabels',[], 'TickLength',[0 0],'XMinorgrid','on','YMinorgrid','on',...
        'gridcolor','w','gridalpha',.9,'minorgridlinestyle','--','minorgridalpha',.5,'MinorGridColor','w')
count = 0; image_h = [];
for i = 1:8:size(Images,1)-1
    image_h = [image_h, axes('Units','pixels','Pos',[pos1(1)+(pos1(3)/16)*count pos1(2)+pos1(4)+2 pos1(3)/16 pos1(4)/16])];
    imshow(Images{i,3})
    image_h = [image_h, axes('Units','pixels','Pos',[pos1(1)+(pos1(3)/16)*count pos1(2)+pos1(4)+2+pos1(4)/16 pos1(3)/16 pos1(4)/16])];
    imshow(Images{i+1,3})
    image_h = [image_h, axes('Units','pixels','Pos',[pos1(1)+pos1(3)+2  pos1(2)+pos1(4)-pos1(4)/16*(count+1)   pos1(3)/16   pos1(4)/16])];
    imshow(Images{i,3})
    image_h = [image_h, axes('Units','pixels','Pos',[pos1(1)+pos1(3)+2+(pos1(3)/16)  pos1(2)+pos1(4)-pos1(4)/16*(count+1)   pos1(3)/16   pos1(4)/16])];
    imshow(Images{i+1,3})
    count = count + 1;
end
% uicontrol('Style','text','units','pixels','position',[pos1(1)+ 0*(pos1(3)/16) pos1(2)+pos1(4)+8+2*(pos1(4)/16) 4*pos1(3)/16 20],'String','Animate','FontSize',14)
% uicontrol('Style','text','units','pixels','position',[pos1(1)+ 4*(pos1(3)/16) pos1(2)+pos1(4)+8+2*(pos1(4)/16) 4*pos1(3)/16 20],'String','Inanimate','FontSize',14)
% uicontrol('Style','text','units','pixels','position',[pos1(1)+ 8*(pos1(3)/16) pos1(2)+pos1(4)+8+2*(pos1(4)/16) 4*pos1(3)/16 20],'String','Animate','FontSize',14)
% uicontrol('Style','text','units','pixels','position',[pos1(1)+12*(pos1(3)/16) pos1(2)+pos1(4)+8+2*(pos1(4)/16) 4*pos1(3)/16 20],'String','Inanimate','FontSize',14)
% uicontrol('Style','text','units','pixels','position',[pos1(1)+ 0*(pos1(3)/16) pos1(2)+pos1(4)+34+2*(pos1(4)/16) 8*pos1(3)/16 20],'String','Drawing','FontSize',14)
% uicontrol('Style','text','units','pixels','position',[pos1(1)+ 8*(pos1(3)/16) pos1(2)+pos1(4)+34+2*(pos1(4)/16) 8*pos1(3)/16 20],'String','Picture','FontSize',14)
text('parent',ax1,  'String','Animate',   'position',[ 2.5 -2.1 0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Inanimate', 'position',[ 6.5 -2.1 0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Animate',   'position',[10.5 -2.1 0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Inanimate', 'position',[14.5 -2.1 0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Drawing',   'position',[ 4.5 -3   0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Picture',   'position',[12.5 -3   0], 'FontSize',14,  'HorizontalAlignment','center')
text('parent',ax1,  'String','Animate',   'position',[19.1  2.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
text('parent',ax1,  'String','Inanimate', 'position',[19.1  6.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
text('parent',ax1,  'String','Animate',   'position',[19.1 10.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
text('parent',ax1,  'String','Inanimate', 'position',[19.1 14.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
text('parent',ax1,  'String','Drawing',   'position',[20    4.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
text('parent',ax1,  'String','Picture',   'position',[20   12.5 0], 'FontSize',14,  'HorizontalAlignment','center', 'Rotation',-90)
time_h = text('parent',ax1,'String',sprintf('%3.0f ms',TimeVec(1)*1000),'position',[8.5 17.1 0],'FontSize',16,'HorizontalAlignment','center');

% Plot MDS
pos2 = [posfig(3)-pos1(3)-pos1(1) pos1(2) pos1(3) pos1(4)];
ax2 = axes('Units','pixels','Pos',pos2);
if(plotpictures)
    scale_mds = 100;
    pl_mds = scatter(MDS_Mat(:,mds_dim1,1).*scale_mds,MDS_Mat(:,mds_dim2,1).*scale_mds);
    %set(ax2,'Ylim',[-30 30],'Xlim',[-30 30])
    pbaspect(ax2,[1 1 1])
    set(pl_mds,'Marker','none')
    for m = 1:size(MDS_Mat,1)
        imExMDS{m,3} = image('CData', imExMDS{m,2}, 'XData', scale_mds.*MDS_Mat(m,mds_dim1,1)+[-size(imExMDS{m,2},2)/50 size(imExMDS{m,2},2)/50], 'YData', scale_mds.*MDS_Mat(m,mds_dim2,1)+[size(imExMDS{m,2},1)/50 -size(imExMDS{m,2},1)/50]);
        imExMDS{m,4} = image('CData', imExMDS{m,1}, 'XData', scale_mds.*MDS_Mat(m,mds_dim1,1)+[-size(imExMDS{m,1},2)/50 size(imExMDS{m,1},2)/50], 'YData', scale_mds.*MDS_Mat(m,mds_dim2,1)+[size(imExMDS{m,1},1)/50 -size(imExMDS{m,1},1)/50]);
    end
else
    scale_mds = 1;
    pl_mds = gscatter(MDS_Mat(:,mds_dim1,1),MDS_Mat(:,mds_dim2,1),X);
    pbaspect(ax2,[1 1 1])
    %legend(X_names); 
end
if(add_centroid)
    hold on
    cent1 = plot(mean(scale_mds.*MDS_Mat(X == 1,mds_dim1,1),1),mean(scale_mds.*MDS_Mat(X == 1,mds_dim2,1),1),'x','Color',[1 0 0],'MarkerSize',15,'linewidth',3); 
    [V,D]= eig(chi2_95perc*cov(scale_mds.*MDS_Mat(X == 1,[mds_dim1 mds_dim2],1)));
    std_err_elipse = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
    elipse1 = patch(std_err_elipse(1,:) + mean(scale_mds.*MDS_Mat(X == 1,mds_dim1,1),1), std_err_elipse(2,:) + mean(scale_mds.*MDS_Mat(X == 1,mds_dim2,1),1), [1 0 0]);
    set(elipse1, 'FaceAlpha',0.1, 'EdgeAlpha',0)
    cent2 = plot(mean(scale_mds.*MDS_Mat(X == 2,mds_dim1,1),1),mean(scale_mds.*MDS_Mat(X == 2,mds_dim2,1),1),'x','Color',[0 1 1],'MarkerSize',15,'linewidth',3);
    [V,D]= eig(chi2_95perc*cov(scale_mds.*MDS_Mat(X == 2,[mds_dim1 mds_dim2],1)));
    std_err_elipse = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
    elipse2 = patch(std_err_elipse(1,:) + mean(scale_mds.*MDS_Mat(X == 2,mds_dim1,1),1), std_err_elipse(2,:) + mean(scale_mds.*MDS_Mat(X == 2,mds_dim2,1),1), [0 1 1]);
    set(elipse2, 'FaceAlpha',0.1, 'EdgeAlpha',0)
    hold off
end
set(ax2,'Xlim',mds_x_limits.*scale_mds,'Ylim',mds_y_limits.*scale_mds)

% Plot Time Series
pos3 = [pos1(1)+pos1(3)/2 50 pos2(1)+pos2(3)/2 - (pos1(1)+pos1(3)/2) pos1(2)-130];
ax3 = axes('Units','pixels','Pos',pos3);
hold on
dat1 = nanmean(ts_data(:,:,1),1);
dat2 = nanmean(ts_data(:,:,2),1);
SEM1 = nanstd(ts_data(:,:,1),0,1)./sqrt(size(ts_data,1));
SEM2 = nanstd(ts_data(:,:,2),0,1)./sqrt(size(ts_data,1));
fill([TimeVec fliplr(TimeVec)],[dat1 fliplr(dat1 + SEM1)],'b','FaceAlpha',0.3,'EdgeAlpha',0);
fill([TimeVec fliplr(TimeVec)],[dat1 fliplr(dat1 - SEM1)],'b','FaceAlpha',0.3,'EdgeAlpha',0);
fill([TimeVec fliplr(TimeVec)],[dat2 fliplr(dat2 + SEM2)],'r','FaceAlpha',0.3,'EdgeAlpha',0);
fill([TimeVec fliplr(TimeVec)],[dat2 fliplr(dat2 - SEM2)],'r','FaceAlpha',0.3,'EdgeAlpha',0);
h1 = plot(TimeVec, dat1,'b','linewidth',2);
h2 = plot(TimeVec, dat2,'r','linewidth',2);
xlim([TimeVec(1) TimeVec(end)])
ylimits = get(ax3,'ylim');
time_line = plot([TimeVec(1) TimeVec(1)],[ylimits(1) ylimits(2)],'color',[1 0 1 0.3],'linewidth',2.5);
hold off
set(ax3,'ylim',ylimits)
%lg = legend([h1 h2], {Dat_names{dt(1)},Dat_names{dt(2)}}); legend boxoff; set(lg,'FontSize',14)

set([ax1 ax2 ax3],'units','norm')
set(image_h,'units','norm')
set(clb,'units','norm')

scale = 30;
% Loop over Time Points
for i = 2:length(TimeVec)
    set(time_h,'String',sprintf('%3.0f ms',TimeVec(i)*1000))
    set(time_line,'XData',[TimeVec(i) TimeVec(i)])
    set(pl_im,'CData',squeeze(RSA_Mat(i,:,:)))
    if(plotpictures)
        for m = 1:size(MDS_Mat,1)
            set(imExMDS{m,3},'XData',scale_mds.*MDS_Mat(m,mds_dim1,i)+[-1 1]*(size(imExMDS{m,2},2)/scale), 'YData',scale_mds.*MDS_Mat(m,mds_dim2,i)+[1 -1])
            set(imExMDS{m,4},'XData',scale_mds.*MDS_Mat(m,mds_dim1,i)+[-1 1]*(size(imExMDS{m,1},2)/scale), 'YData',scale_mds.*MDS_Mat(m,mds_dim2,i)+[1 -1])
        end
        yfac = diff(get(ax2,'ylim'))/diff(get(ax2,'xlim'));
        for m = 1:size(MDS_Mat,1)
            set(imExMDS{m,3},'YData',scale_mds.*MDS_Mat(m,mds_dim2,i)+[1 -1]*yfac*size(imExMDS{m,2},1)/scale)
            set(imExMDS{m,4},'YData',scale_mds.*MDS_Mat(m,mds_dim2,i)+[1 -1]*yfac*size(imExMDS{m,1},1)/scale)
        end
    else
        set(pl_mds(1), 'XData', MDS_Mat(X == 1,mds_dim1,i), 'YData', MDS_Mat(X == 1,mds_dim2,i))
        set(pl_mds(2), 'XData', MDS_Mat(X == 2,mds_dim1,i), 'YData', MDS_Mat(X == 2,mds_dim2,i))
    end
    if(add_centroid)
        set(cent1, 'XData', mean(scale_mds.*MDS_Mat(X == 1,mds_dim1,i),1), 'YData', mean(scale_mds.*MDS_Mat(X == 1,mds_dim2,i),1))
        [V,D]= eig(chi2_95perc*cov(scale_mds.*MDS_Mat(X == 1,[mds_dim1 mds_dim2],i)));
        std_err_elipse = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
        set(elipse1, 'XData', std_err_elipse(1,:) + mean(scale_mds.*MDS_Mat(X == 1,mds_dim1,i),1), 'YData', std_err_elipse(2,:) + mean(scale_mds.*MDS_Mat(X == 1,mds_dim2,i),1))
        set(cent2, 'XData', mean(scale_mds.*MDS_Mat(X == 2,mds_dim1,i),1), 'YData', mean(scale_mds.*MDS_Mat(X == 2,mds_dim2,i),1))
        [V,D]= eig(chi2_95perc*cov(scale_mds.*MDS_Mat(X == 2,[mds_dim1 mds_dim2],i)));
        std_err_elipse = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
        set(elipse2, 'XData', std_err_elipse(1,:) + mean(scale_mds.*MDS_Mat(X == 2,mds_dim1,i),1), 'YData',std_err_elipse(2,:) + mean(scale_mds.*MDS_Mat(X == 2,mds_dim2,i),1))
    end
    pause(0.3)
end
