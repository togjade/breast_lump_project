addpath('/home/togzhan/togzhan_catkin_ws/src/imu_files/imu_libraries/madgwick_algorithm_matlab/');

addpath('/home/togzhan/togzhan_catkin_ws/src/imu_files/imu_libraries/madgwick_algorithm_matlab/quaternion_library');
% cd /home/togzhan/togzhan_catkin_ws/src/imu_files/msg/
addpath('/home/togzhan/togzhan_catkin_ws/src/matlab_msg_gen_ros1/glnxa64/install/m')
savepath
%%
% folderpath = '/home/togzhan/togzhan_catkin_ws/src';
% %fullfile('togzhan_catkin_ws','src')
% rosgenmsg
%%
clc;clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all; 
setenv('ROS_MASTER_URI','http://10.1.71.208:11311/');
setenv('ROS_IP','10.1.71.208');
rosinit

pub1 = rospublisher('imu__data', 'imu_files/imu_data');
msg1 = rosmessage('imu_files/imu_data');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('quaternion_library');      % include quaternion library
AHRS1 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);
AHRS2 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);
AHRS3 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);
AHRS4 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);
AHRS5 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);
AHRS6 = MadgwickAHRS('SamplePeriod', 1/100, 'Beta', 10);   %sqrt(3.0 / 4.0)*14*pi/180=0.21
AHRS=[AHRS1, AHRS2, AHRS3, AHRS4, AHRS5, AHRS6];
quaternion=[];
%
quat_ref=[0.1065    0.1270    0.6391   -0.7511];
% drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0,'linewidth',3);  
%
str1 = '/dev/ttyUSB';
str2 = 
% Create serial object for Arduino 
s = serial('/dev/ttyUSB0','BaudRate',1000000); % change the COM Port number as needed (Lowest)
%s.InputBufferSize = 4096; % read only one byte every time 
s.InputBufferSize = 220;
try 
    fopen(s); 
catch err 
    fclose(instrfind);
    error('Make sure you select the correct COM Port where the Arduino is connected.');
end
set(s,'Terminator','CR/LF');

data_array=[];
kk = 1;
qq = [];
rpy_old = zeros(1,6);
while 1
tic;

fwrite(s,'a');
data=fgetl(s);
while length(data)~=110
    fwrite(s,'a');
    data=fgetl(s);

end
values=zeros(55,1);
for i=1:55
    values(i)=typecast([uint8(data(2*(i-1)+2)) uint8(data(2*(i-1)+1))],'int16');
end
data_line=conversion_My(values);
quaternion(:,end+1, 1) = zeros(4,1,1);
q = {};

for j=1:6
    AHRS(j).Update(data_line(9*(j-1)+1:9*(j-1)+3)', data_line(9*(j-1)+4:9*(j-1)+6)', data_line(9*(j-1)+7:9*(j-1)+9)');	% gyroscope units must be radians
    quaternion(:,end, j) = AHRS(j).Quaternion;
    q{j,kk} = AHRS(j).Quaternion;
end

kk=kk+1;
% 
q = cell2mat(q);
q_base_inv = quaternConj(q(1, :));
q = q_base_inv.*q();
rpy = quat2eul(q);

qq(end+1, :)  = rpy(:, 2)';
% rpy*180/pi
%     if(size(qq,1)>2)
%         qq(1,:) = [];
%         rpy_avg = mean(qq);
%         rpy_avg;
%         rpy;
%         rpy_old;
%         for l = 1:6
%            if(abs(rpy(l,2) - rpy_avg(l)) > 3/180*pi)
%                 rpy_old(1,l) = rpy(l, 2)+30*pi/180;
%            end 
%         end
%         msg1.ImuMatData =  rpy_old;
%         send(pub1, msg1);
%     end
msg1.ImuMatData =  rpy;
send(pub1, msg1);


while 1                 % to keep synchronized
       if toc>0.025
           break
       end
       java.lang.Thread.sleep(0.001*1000); %pause for 1ms
end

waitfor(rpy);

end
rosshutdown;
%%
% 
% function res = highPass(z, z_, out)
%     T = 0.2;
%     dt = 0.02;
%     res = (out + z - z_)*T/(T + dt);
%     return 
% end
% function a = to_grasp(data)
%     sum_pres = abs(sum(data));
%     while sum_pres < 3/180*pi
%         sum_pres = abs(sum(data));
%     if (sum_pres > 3/180*pi)
%        a = 1;
%        return 
%     end
%     end
% end
