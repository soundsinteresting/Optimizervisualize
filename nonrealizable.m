function nonrealizable()
maxstep=10000000;
ave=1;
xi=zeros(ave,maxstep);
i=1;
figure;
bg=0;
hold on;
[~,~,xi(i,:)]=quadraticexample2(maxstep,0.0, 0.999,100,-0.00,0.1,false,false);
%plot(1+bg*maxstep:maxstep,log10(abs(xi(1,1+bg*maxstep:maxstep)+10^(-20))),'Color',[1 0.1 0],'LineWidth',1);
plot(1+bg*maxstep:maxstep,xi(1,1+bg*maxstep:maxstep),'Color',[1 0.1 0],'LineWidth',1);
[~,~,xi(i,:)]=quadraticexample2(maxstep,0.0, 0.999,100,-0.00,0.1,true,false);
%plot(1+bg*maxstep:maxstep,log10(abs(xi(1,1+bg*maxstep:maxstep)+10^(-20))),'Color',[0.1 1 0],'LineWidth',1);
plot(1+bg*maxstep:maxstep,xi(1,1+bg*maxstep:maxstep),'Color',[0.1 1 0],'LineWidth',1);
[~,~,xi(i,:)]=quadraticexample2(maxstep,0.0, 0.999,100,-0.00,0.1,false,true);
%plot(1+bg*maxstep:maxstep,log10(abs(xi(1,1+bg*maxstep:maxstep)+10^(-20))),'Color',[0 0.1 1],'LineWidth',1);
plot(1+bg*maxstep:maxstep,xi(1,1+bg*maxstep:maxstep),'Color',[0 0.1 1],'LineWidth',1);
ylim([-1 1])
legend('RMSprop','AMSgrad','SGD');
xlabel('number of iteration','Fontsize',12);
ylabel('x-x^*','Fontsize',12);
%ylabel('x','Fontsize',12);
hold off;
slope=(log10(abs(xi(maxstep)))-log10(abs(xi(maxstep/10))))/maxstep*10/9

%plot_map()
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
grad=100*(xk(1)-dev)^2+10*(xk(1)-10/9*dev)^2;%startp^2;
xi=xk(1);
order=randperm(bs);
    for k=1:maxstep
%         if mod(k,bs)==0
%             order=randperm(bs);
%         end
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
        xk(k+1)=xk(k)-eta*(ceil(k/10)*10)^(-0.5)*momentum/sqrt(grad);   
        if SGD
            xk(k+1)=xk(k)-eta*(ceil(k/10)*10)^(-0.5)*momentum; 
        end
        xi=xk(k+1);
    end
    xk=xk(1:maxstep);
end
