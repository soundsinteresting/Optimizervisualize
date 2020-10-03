function realizable()
maxstep=1000000;
ave=1;
xi=zeros(ave,maxstep);

for i=1:ave
%non realizable problem, first false is AMSGrad, second false is SGD
%[~,~,xi(i,:)]=quadraticexample2(maxstep,0.0, 0.999,10000,0.01,0.1,false,false);
%realizable problem
[~,~,xi(i,:)]=quadraticexample2(maxstep,0.0, 0.999,0,0.01,0.1,false,false);
end
xi=mean(xi,1);
figure;
bg=0;
plot(1+bg*maxstep:maxstep,log10(abs(xi(1,1+bg*maxstep:maxstep)+10^(-20))),'Color',[0.1 1 0],'LineWidth',1);
hold on;

xlabel('number of iteration','Fontsize',12);
ylabel('lg(x-x^*)','Fontsize',12);
%ylabel('x','Fontsize',12);
hold off;
slope=(log10(abs(xi(maxstep)))-log10(abs(xi(maxstep/10))))/maxstep*10/9

%plot_map()
end

function plot_map()
maxstep=20000;
a=21;
b=30;
beta2_conv=zeros(a*b);
beta2_div=zeros(a*b);
C_conv=zeros(a*b);
C_div=zeros(a*b);
temp=12;
st_c=1;
st_d=1;
    for i=1:20
        for C=1:31
            beta2=1-10^(-i/temp);
            xi=zeros(maxstep);
            [~,~,xi]=reddiexample(maxstep,C,beta2);
            terminal=log10(abs(xi(maxstep)+1));
            if terminal<-2
                beta2_conv(st_c)= i/temp;
                C_conv(st_c)=C;
                st_c=st_c+1;
            else
                beta2_div(st_d)=i/temp;
                C_div(st_d)=C;
                st_d=st_d+1;
            end                
        end
        i
    end
    beta2_conv=beta2_conv(1:st_c);
    C_conv=C_conv(1:st_c);
    beta2_div=beta2_div(1:st_d);
    C_div=C_div(1:st_d);
    plot(beta2_conv,C_conv,'o');
    hold on;
    plot(beta2_div,C_div,'+');
    xlabel('-lg(1-beta_2)','Fontsize',12);
    ylabel('C','Fontsize',12);
    legend('convergent', 'divergent');
    hold off;
end
function [vk,gk,xk]=reddiexample(maxstep,C,beta2)
%C=10;
eta=1;
beta1=0;
%beta2=1-1/C^2;
1-1/C^2;

%beta2=0.95;
xk=zeros(maxstep,1);
vk=zeros(maxstep,1);
gk=zeros(maxstep,1);
%grad=0;
momentum=0;
j=0;
xk(1)=100;
grad=xk(1)*C;
(1/C^2)^(2/(C-2));
    for k=1:maxstep
        %i=ceil(3*rand());
        j=j+1;
        i=mod(j,C);
        %i=ceil(C*rand());
        if i<=1
            gr=C;
        else
            gr=-1;
        end
        gk(k)=gr^2;
        grad=grad*beta2+(1-beta2)*gr^2;
        vk(k)=grad;
        momentum=momentum*beta1+(1-beta1)*gr;
        xk(k+1)=xk(k)-eta*momentum/sqrt(grad)*(k)^(-0.5);
        if xk(k+1)>1
            xk(k+1)=1;
        end
        if xk(k+1)<-1
            xk(k+1)=-1;
        end            
    end
    xk=xk(1:maxstep);
end

function [vk,gk,xk]=quadraticexample2(maxstep,beta10, beta20,dev,startp, step,AMSGrad, SGD)
bs=10;
eta=step;
beta1=beta10;
xk=zeros(maxstep,1);
vk=zeros(maxstep,1);
gk=zeros(maxstep,1);

momentum=0;
xk(1)=startp;
grad=0;%100*(xk(1)-dev)^2+10*(xk(1)-10/9*dev)^2;%startp^2;
xi=xk(1);
order=randperm(bs);
    for k=1:maxstep
        if mod(k,bs)==0
            order=randperm(bs);
        end
        %c=order(mod(k,bs)+1);
        c=k;
        if mod(c,bs)==0
            gr=10*(xi-dev);
        else
            gr=-(xi-10/9*dev);
            %gr=(xi-dev);
        end
        beta2=1-(1-beta20)*k^(-0);
        %beta2=1;
        newgrad=grad*beta2+(1-beta2)*gr^2;
        if AMSGrad 
            if newgrad>grad
                grad=newgrad;
            end
        else
            grad=newgrad;
        end
        
        gk(k)=gr;
        vk(k)=grad;
        momentum=momentum*beta1+(1-beta1)*gr;
        xk(k+1)=xk(k)-eta*k^(-0.5)*momentum/sqrt(grad);   
        if SGD
            xk(k+1)=xk(k)-eta*k^(-0.5)*momentum; 
        end
        xi=xk(k+1);
    end
    xk=xk(1:maxstep);
end
