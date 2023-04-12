%% conversion of raw IMU values to AHRS ready
function [values]=conversion_My(values)
data_line=values;

m_min=zeros(1,3,6);
m_min(:,:,1)=[ -4927,  -3709, -13970];
m_min(:,:,2)=[ -4810,  -3857,  -3813];
m_min(:,:,3)=[ -5489,  -4393,  -3832];
m_min(:,:,4)=[ -3849,  -5409,  -5853];
m_min(:,:,5)=[ -5226,  -2461,  -6537];
m_min(:,:,6)=[-5426,  -3206,  -3522];
m_max=zeros(1,3,6);
m_max(:,:,1)=[+2939,  +4924,  -1954];
m_max(:,:,2)=[+3298,  +2990,  +7104];
m_max(:,:,3)=[ +3054,  +2822,  +6923];
m_max(:,:,4)=[  +4284,  +1756,  +5429];
m_max(:,:,5)=[ +4647,  +5530,  +4353];
m_max(:,:,6)=[ +3446,  +5428,  +5988];



g_offset=zeros(1,3,6);


g_offset(:,:,1) =[ -44.2500000000000	-636.125000000000	-402.175000000000];
g_offset(:,:,2) =[ -62.1375000000000	-610.125000000000	-400.075000000000];
g_offset(:,:,3) =[-60.3250000000000	-605.987500000000	-390.487500000000];
g_offset(:,:,4) =[-87.0625000000000	-595.900000000000	-388.012500000000];
g_offset(:,:,5) =[-17.7750000000000	-639.725000000000	-393.187500000000];
g_offset(:,:,6) =[-36.7875000000000	-672.975000000000	-397.350000000000];



for j=1:6                           % dps(degrees per second)
    % gyro: 2000 dps full scale, normal power mode, all axes enabled, 100 Hz ODR 12.5Hz Bandwith
    values(9*(j-1)+1:9*(j-1)+3)=(data_line(9*(j-1)+1:9*(j-1)+3)'-g_offset(:,:,j))*0.07 * (pi/180);   %  rad/s
    values(9*(j-1)+4:9*(j-1)+6)=data_line(9*(j-1)+4:9*(j-1)+6) * 0.000244;  % accelerom: 8 g full scale, 16bit representation, all axes enabled, 100 Hz ODR
    values(9*(j-1)+7:9*(j-1)+9)=((data_line(9*(j-1)+7:9*(j-1)+9)'-m_min(:,:,j))./(m_max(:,:,j)-m_min(:,:,j)) * 2 -1 );%* 4/65535;  % magnetom 
    % compass.m_min = (LSM303::vector<int16_t>){-32767, -32767, -32767};
    % compass.m_max = (LSM303::vector<int16_t>){+32767, +32767, +32767};
                                    %gyroscope units must be radians
end




end


% for j=1:6                           % dps(degrees per second)
%     % gyro: 2000 dps full scale, normal power mode, all axes enabled, 100 Hz ODR 12.5Hz Bandwith
%     values(9*(j-1)+1:9*(j-1)+3)=(data_line(9*(j-1)+1:9*(j-1)+3)'-g_offset(:,:,j)) * (pi/180)*(8.75 / 1000);   %  rad/s
%     values(9*(j-1)+4:9*(j-1)+6)=data_line(9*(j-1)+4:9*(j-1)+6)*(0.061 / 1000);% * 8/32767;  % accelerom: 8 g full scale, 16bit representation, all axes enabled, 100 Hz ODR
%     values(9*(j-1)+7:9*(j-1)+9)=(data_line(9*(j-1)+7:9*(j-1)+9)'-(m_min(:,:,j) + m_max(:,:,j)) / 2 )*(0.00014615609);% 4/65535;  % magnetom 
%     % compass.m_min = (LSM303::vector<int16_t>){-32767, -32767, -32767};
%     % compass.m_max = (LSM303::vector<int16_t>){+32767, +32767, +32767};
%                                     %gyroscope units must be radians
% end