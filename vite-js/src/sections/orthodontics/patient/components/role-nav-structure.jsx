import './role-nav-structure.css';

import React from 'react';
import { useNavigate } from 'react-router-dom';

import { getFrontendUrl } from 'src/utils/url-helpers';

export default function RoleNavStructure({ onSelect, activeKey, patientId }) {
  const navigate = useNavigate();

  // Items mapped to patient tabs order
  // Updated to user's requested order and icons (using available navbar masks)
  const items = [
    { key: 'general', label: 'مشخصات', href: '#general', icon: 'user', iconType: 'svg' },
    { key: 'diagnosis', label: 'تشخیص', href: '#diagnosis', icon: 'ai', iconType: 'png' },
    { key: 'cephalometric', label: 'سفالومتری', href: patientId ? `${getFrontendUrl()}/dashboard/orthodontics/patient/${patientId}/analysis` : '#cephalometric', icon: 'cephalometry', iconType: 'png' },
    { key: 'intra-oral', label: 'داخل دهان', href: '#intra-oral', icon: 'lips', iconType: 'png' },
  ];

  const handleClick = (e, key, href) => {
    // If href is an absolute URL (external), navigate directly
    if (href && /^https?:\/\//.test(href) && !href.includes(window.location.origin)) {
      // External URL - let the browser navigate
      window.location.href = href;
      return;
    }
    // If href is a full path (internal route), use navigate for SPA
    if (href && href.startsWith('/')) {
      e.preventDefault();
      navigate(href);
      return;
    }
    // Otherwise, it's a hash link for internal tabs
    e.preventDefault();
    onSelect?.(key);
  };

  return (
    <div className="role-nav-section" aria-label="role navigation demo">
      <div className="role-nav-card">
        <div className="role-nav-content">
          <nav className="role-nav horizontal" aria-label="primary navigation">
            <ul className="role-nav-list horizontal">
              {items.map((it) => {
                const isActive = activeKey === it.key;
                return (
                  <li className="role-nav-item" key={it.key}>
                    <a
                      tabIndex={0}
                      aria-label={it.label}
                      href={it.href}
                      className={`role-nav-link ${isActive ? 'active' : ''}`}
                      onClick={(e) => handleClick(e, it.key, it.href)}
                    >
                      <span className="role-nav-icon">
                        {it.iconType === 'png' ? (
                          <img 
                            src={`/assets/icons/${it.icon}.png`} 
                            alt={it.label}
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          />
                        ) : (
                        <span className={`role-nav-icon-${it.icon}`} />
                        )}
                      </span>
                      <span className="role-nav-label">{it.label}</span>
                      <span className="role-nav-link-overlay" />
                    </a>
                  </li>
                );
              })}
            </ul>
          </nav>
        </div>
      </div>
    </div>
  );
}
