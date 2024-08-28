%%
close all;
clear all;
%% parameters defined
sample_rate = 0.001;
simulate_time = 200;
numSubcarriers = 1;
bandwidth = 200e6;
t = 0:sample_rate:simulate_time - sample_rate;
num_ue = 20;
radius = 1e5;
shotNum = length(t);
userStruct = struct('pow', [], 'angle', []);
results = repmat(userStruct, num_ue, 1);
%% Setting general parameters
s = qd_simulation_parameters;                           % New simulation parameters
s.center_frequency = 2e10;                             % 2.4 GHz center frequency
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.sample_density = 2;                                 % Minimum possible sample density
sat_array = qd_arrayant('parabolic', 0.25, 20e9,[],5,1,[], 38.5 );   % Set1, LEO, KA, DL
ue_array = qd_arrayant('parabolic',  0.6, 20e9,[],5,1,[], 39.7 );     % VSAT, KA, DL
satellite = qd_satellite( 'custom', qd_satellite.R_e + 1300, 0, 63.4, -28.8, 44.55, 0);
%sat = satellite.init_tracks( [-5,38.85], t );   	        % UE reference position on Earth
sat = satellite.init_tracks( [3,46.85], t );
sat.name = 'BS satellite';
theta = 2 * pi * rand(1, num_ue);
r = radius * (rand(1, num_ue));
x_coords = r .* cos(theta);
y_coords = r .* sin(theta);

ue = qd_track('linear', 0);
%ue.interpolate('time', sample_rate, [], [], 1);
ue.no_snapshots = shotNum;
mov_pro1 = t;
mov_pro2 = 1:1:shotNum;
ue.movement_profile = [mov_pro1;mov_pro2];
ue.interpolate('snapshot',sample_rate,ue.movement_profile,[],1)
%l.tx_array = sat_array;
%l.rx_array = ue_array;


%% Basestation Satellite
parfor i = 1:num_ue
    l = qd_layout( s );                                     % New layout
    l.tx_track = sat;                                    % Assign Rx track 2
    l.tx_array = sub_array( sat_array,1 );
    l.rx_track = ue;
    l.rx_array = sub_array( ue_array,1 ); 
    l.rx_track.initial_position = [x_coords(i);y_coords(i);1.5];
    l.rx_track.name = ['UE' num2str(i)];
    l.set_scenario('QuaDRiGa_NTN_Rural_LOS');           % Static transmitter
%% Orientation
    rx_pos = l.rx_track.initial_position + l.rx_track.positions;
    tx_pos = l.tx_track.initial_position + l.tx_track.positions; 
    for x = 1:length(t)
        rt = tx_pos(:,x) - rx_pos(:,x);
        rt = rt / norm(rt);                             % Normalize to unit length
        tt = [0;0;0] - tx_pos(:,x);                % Satellite point to [0,0,0]
        tt = tt / norm(tt);
        l.rx_track.orientation(1,x) = 0;
        l.rx_track.orientation(2,x) = asin( rt(3) );               	% Calculate UE tilt angle
        l.rx_track.orientation(3,x) = atan2( rt(2),rt(1) );        	% Calculate UE heading angle
        l.tx_track.orientation(2,x) = asin( tt(3) );
        l.tx_track.orientation(3,x) = atan2( tt(2),tt(1) );
    end
    l.update_rate = sample_rate;
    c = l.get_channels;
    %% Path gain
% Now we plot the path-gain for the 3 generated channels. As Car1 moves away from the BS, its PG
% decreases from roughly -40 dB to about -100 dB. Likewise, the PG of Car2 increases. The PG of the
% Car1-Car2 channel starts at a low vale and increases until the cars pass each other at about 4.8
% seconds simulation time. Then, the PG decreases again.
    
    time = ( 0 : c(1,1).no_snap-1 ) * sample_rate;        % Time axis in seconds
    pg = [ c(1,1).par.pg ]; % The path-gain values
    h = c(1,1).fr(bandwidth, numSubcarriers); % 1 x NbaseAnt x numSubcarriers x sequenceLength
    %pow  = 10*log10(sum(abs(squeeze(c.coeff(:,:,:,:))).^2,1));
    pow = 10*log10(abs(squeeze(h(:,:,:,:))).^2);
    %set(0,'DefaultFigurePaperSize',[14.5 4.7])              % Change paper Size
    %figure('Position',[ 100 , 100 , 760 , 400]);            % New figure
    %plot(time,pow','-','Linewidth',2)                        % Plot target PG
    %title('Path Gain / Time'); 
    %xlabel('Time/s'); ylabel('Path Gain/dB');
    %axis([0,max(time),min(pow(:))-3,max(pow(:))+3]); grid on;
    %legend('satellite - UT');
    %% Save the channel to channel.mat
    angle = l.rx_track.orientation(2,:);
    results(i).pow = pow';
    results(i).angle = angle;
end
save('results-1300.mat','results')
%%
% Now, the scenarios are assigned. The BS-Car links use the default 3GPP Urban-Microcell parameters.
% For Car-Car channels, we use initial Urban-Device-to-Device parameters. Those have not been
% confirmed by measurements yet. Since "Car1" acts as both, a transmitter and a receiver, we also
% need to remove the "Car1-Car1" link from the channel list. Lastly, a plot of the scenario is
% created showing the BS coverge and the trajectories.


%{
ls = qd_layout( s );                                     % New layout
ls.tx_position = l.tx_position;                                    % Assign Rx track 2
ls.tx_array = l.tx_array;
ls.tx_track.orientation = l.tx_track.orientation(:,1);
ls.rx_position = l.rx_position;
ls.rx_track(1,1).orientation = l.rx_track(1,1).orientation(:,1);
ls.rx_track(1,2).orientation = l.rx_track(1,2).orientation(:,1);
ls.rx_track(1,3).orientation = l.rx_track(1,3).orientation(:,1);
ls.rx_array = l.rx_array;

[ map,x_coords,y_coords ] = ls.power_map( 'QuaDRiGa_NTN_Urban_LOS',...
        'quick' ,100, -80000, 40000, -80000, 20000, 1.5);                % Genearte coverage maps for each beam

p = map{1};
p = squeeze(max(max(p,[],3),[],4));
P_db = 10*log10(p);   
    
ls.visualize([],[],0);                                  % Show Sat and MT positions on the map
hold on
imagesc( x_coords, y_coords, P_db );                      % Plot the Geometry Factor
hold off
                                   % Show BS and MT positions on the map
%}
%% Calculate channel coefficients
% The following command calculates the channel coefficients once per millisecond. The status update
% is is shown on the command line. This involves the following steps: 
% p
% * Interpolation of the tracks to match the sample density. This avoids unnecessary computations
%   but makes sure, that the Doppler profile is completely captured. At 2.4 GHz carrier frequency,
%   250 m track length, and a sample density of 2.1, 8407 snapshots are needed.
% * Generation of channel builder objects and assigning track segments to builders.
% * Generation of large and small-scale-fading parameters, including spatial consistency.
% * Generation of drifting channel coefficients for each track-segment.
% * Merging of channel segments, including modeling the birth and death of scattering clusters.
% * Interpolation of channel coefficients to match the sample rate. This generates 9001 snapshots at
%   the output. 

%l.visualize([],[],0);

