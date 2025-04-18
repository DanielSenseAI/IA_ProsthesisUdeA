function [i_marks,e_marks]=markDetFlow(signal)

%Algorithm for  selection  of  inspiratory and expiratory marks from a
%Flow signal. This algorithm assumes that any inspiration must exceed
%a threshold of 0.1.  The input is the flow signal to which the marks want to be
%detected, and it gives as output the the ispiratory and expiratory marks,
%Ex: [inspiratory_marks, expiratory_marks]=markdet(flow_signal)
%
%Last reviewed: 18-08-2010
%By: Alejandro Camacho Laverde

non_zero=find(signal);
counter=1;

pseudo_marks=zeros(length(non_zero),1);

for i=1:length(non_zero)-1
    
    a=signal(non_zero(i));
    b=signal(non_zero(i+1));
    
    sign_criteria=sign(a/b);
    
    if sign_criteria==-1
        
        pseudo_marks(counter)=non_zero(i+1);
        counter=counter+1;
        
    end
    
end

pseudo_marks=pseudo_marks(1:counter-1);
phase=zeros(length(pseudo_marks)-1,1);

% Evaluation of the inspiratory phase; if the result of the following code
% is -1, the current sample belongs to an inspiratory phase, if the result
% is 1, it belongs to an expiratory phase.

for i=1:length(pseudo_marks)-1
    
    a=signal(pseudo_marks(i));
    b=signal(pseudo_marks(i+1));
    phase(i)=sign(b-a);
    
    
end

%Now, we determine which of the supossed marks are actual marks, utilizing
%the energy method, proposed by

E_exp=0.5;
E_ins=0.5;
i_marks=zeros(length(pseudo_marks)-1,1); % Inspiratory marks
e_marks=zeros(length(pseudo_marks)-1,1); % Expiratory marks
i_marks_counter=1;
e_marks_counter=1;

for j=1:length(pseudo_marks)-1
    
    
    
    E_sum=0;
    
    if phase(j)==1
        
        threshold= E_exp;
        
        
    else
        threshold=E_ins;
        
    end
    
    
    for i=pseudo_marks(j):pseudo_marks(j+1)-1
        
        E_sum=signal(i)^2+ E_sum;
        
        
        
        if E_sum>=threshold
            
            if phase(j)==-1;
                
                if signal(pseudo_marks(j)-1)==0
                    
                    i_marks(i_marks_counter)=pseudo_marks(j)-1;
                    
                else
                    
                    i_marks(i_marks_counter)=pseudo_marks(j);
                    
                end
                
                
                i_marks_counter=i_marks_counter+1;
                
            else
                
                if signal(pseudo_marks(j)-1)==0
                    
                    e_marks(e_marks_counter)=pseudo_marks(j)-1;
                    
                else
                    
                    e_marks(e_marks_counter)=pseudo_marks(j);
                    
                end
                
                
                e_marks_counter=e_marks_counter+1;
                
                
            end
            
            break;
            
        end
        
    end
    
end

i_marks=i_marks(1:i_marks_counter-1);
e_marks=e_marks(1:e_marks_counter-1);



end
